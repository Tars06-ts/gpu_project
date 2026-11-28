#include <climits>
#include <cuda_runtime.h>
#include "graph/csr.cuh"
#include "graph/tree.cuh"
#include "graph/types.cuh"
#include "graph/checks.cuh"
#include "SSSP/sssp_update.cuh"

// bitwise OR between two boolean flag arrays
__global__ void d_or_flags(bool* dst, const bool* src, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n && src[tid]) {
        dst[tid] = true;
    }
}

// copies raised flags from src_flags into dst_flags for vmfb compaction
struct IdentityMarkOp {
    __device__ void operator()(bool* src_flags, int n_op, bool* dst_flags, int n_f, int tid) const {
        if (tid < n_f && src_flags[tid]) {
            dst_flags[tid] = true;
        }
    }
};

// Operate on an edge being deleted and finds 1st level deletion affected vertices
struct ProcessDelOp {
    TreeView T;

    __device__ void operator()(Edge* op, int n_op, bool* flags, int n_f, int tid) const {
        int u = op[tid].u;
        int v = op[tid].v;
        if (v < 0 || v >= n_f || v == T.source) return;

        if (T.parent[v] == u) {
            T.dist[v] = INF_DIST;
            T.parent[v] = -1;
            T.exist[v] = false;
            flags[v] = true;
        }
    }
};

// Operate on an edge being inserted and finds 1st level insertion affected vertices
struct ProcessInsOp {
    TreeView T;

    __device__ void operator()(Edge* op, int n_op, bool* flags, int n_f, int tid) const {
        int u = op[tid].u;
        int v = op[tid].v;
        int w = op[tid].weight;
        if (u < 0 || u >= n_f || v < 0 || v >= n_f || v == T.source) return;

        int du = T.dist[u];
        if (du != INF_DIST) {
            int new_dist = du + w;
            if (new_dist < T.dist[v]) {
                atomicMin(&T.dist[v], new_dist);
                if (T.dist[v] == new_dist) {
                    T.parent[v] = u;
                    T.exist[v] = true;
                }
                flags[v] = true;
            }
        }
    }
};

// Operate on a deletion affected vertex and disconnects child neighbors
template<typename GraphViewPtr>
struct DisconnectCOp {
    GraphViewPtr G;
    TreeView T;

    __device__ void operator()(int* op, int n_op, bool* flags, int n_f, int tid) const {
        int u = op[tid];
        if (u < 0 || u >= n_f) return;

        G->for_each_out_neighbor(u, [&] __device__ (int v, int /*w*/) {
            if (v >= 0 && v < n_f && v != T.source && T.parent[v] == u) {
                T.dist[v] = INF_DIST;
                T.parent[v] = -1;
                T.exist[v] = false;
                flags[v] = true;
            }
        });
    }
};

// Operate on an affected vertex to connect disconnected vertices and update neighbors
template<typename GraphViewPtr>
struct ChkNbrOp {
    GraphViewPtr G;
    TreeView T;

    __device__ void operator()(int* op, int n_op, bool* flags, int n_f, int tid) const {
        int x = op[tid];
        if (x < 0 || x >= n_f) return;

        // Pull from incoming neighbors (u -> x)
        if (x != T.source) {
            G->for_each_in_neighbor(x, [&] __device__ (int u, int w) {
                if (u < 0 || u >= n_f) return;
                int du = T.dist[u];
                if (du != INF_DIST) {
                    int cand = du + w;
                    if (cand < T.dist[x]) {
                        atomicMin(&T.dist[x], cand);
                        if (T.dist[x] == cand) {
                            T.parent[x] = u;
                            T.exist[x] = true;
                        }
                        flags[x] = true;
                    }
                }
            });
        }

        // Push to outgoing neighbors (x -> v)
        int dx = T.dist[x];
        if (dx != INF_DIST) {
            G->for_each_out_neighbor(x, [&] __device__ (int v, int w) {
                if (v < 0 || v >= n_f || v == T.source) return;
                int cand = dx + w;
                if (cand < T.dist[v]) {
                    atomicMin(&T.dist[v], cand);
                    if (T.dist[v] == cand) {
                        T.parent[v] = x;
                        T.exist[v] = true;
                    }
                    flags[v] = true;
                }
            });
        }
    }
};

template<typename GraphImpl>
void update_sssp(GraphImpl& graph, Tree& tree, Edge* d_ins, int n_ins, Edge* d_del, int n_del) {
    // Apply batch updates via IGraph's public apply()
    auto* d_G = graph.view();
    if (d_G == nullptr) d_G = graph.to_gpu();

    graph.apply(d_ins, n_ins, d_del, n_del);
    d_G = graph.view();

    TreeView d_T = tree.view();
    if (d_T.dist == nullptr) d_T = tree.to_gpu();

    int n_v = tree.size();
    auto pred = [] __device__ (const bool& flag) { return flag; };

    bool* d_flags = nullptr;
    bool* d_aff_all_flags = nullptr;
    int* d_frontier = nullptr;

    CUDA_CHECK(cudaMalloc(&d_flags, sizeof(bool) * n_v));
    CUDA_CHECK(cudaMalloc(&d_aff_all_flags, sizeof(bool) * n_v));
    CUDA_CHECK(cudaMalloc(&d_frontier, sizeof(int) * n_v));
    CUDA_CHECK(cudaMemset(d_aff_all_flags, 0, sizeof(bool) * n_v));

    // ProcessDel
    CUDA_CHECK(cudaMemset(d_flags, 0, sizeof(bool) * n_v));
    int n_aff_del = 0;
    if (n_del > 0) {
        n_aff_del = graph.vmfb(d_del, n_del, ProcessDelOp{d_T}, d_flags, n_v, pred, d_frontier, n_v);
    }
    if (n_aff_del > 0) {
        CUDA_CHECK(cudaMemcpy(d_aff_all_flags, d_flags, sizeof(bool) * n_v, cudaMemcpyDeviceToDevice));
    }

    // Disconnect subtrees under Aff_del
    while (n_aff_del > 0) {
        CUDA_CHECK(cudaMemset(d_flags, 0, sizeof(bool) * n_v));
        n_aff_del = graph.vmfb(d_frontier, n_aff_del, DisconnectCOp<decltype(d_G)>{d_G, d_T},
                               d_flags, n_v, pred, d_frontier, n_v);
        if (n_aff_del > 0) {
            d_or_flags<<<(n_v + 255) / 256, 256>>>(d_aff_all_flags, d_flags, n_v);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // ProcessIns
    CUDA_CHECK(cudaMemset(d_flags, 0, sizeof(bool) * n_v));
    int n_aff_ins = 0;
    if (n_ins > 0) {
        n_aff_ins = graph.vmfb(d_ins, n_ins, ProcessInsOp{d_T}, d_flags, n_v, pred, d_frontier, n_v);
    }
    if (n_aff_ins > 0) {
        d_or_flags<<<(n_v + 255) / 256, 256>>>(d_aff_all_flags, d_flags, n_v);
        CUDA_CHECK(cudaGetLastError());
    }

    // Union Aff_Alldel U Aff_ins and iteratively relax neighbors
    CUDA_CHECK(cudaMemset(d_flags, 0, sizeof(bool) * n_v));
    int n_aff_all = graph.vmfb(d_aff_all_flags, n_v, IdentityMarkOp{}, d_flags, n_v, pred, d_frontier, n_v);

    while (n_aff_all > 0) {
        CUDA_CHECK(cudaMemset(d_flags, 0, sizeof(bool) * n_v));
        n_aff_all = graph.vmfb(d_frontier, n_aff_all, ChkNbrOp<decltype(d_G)>{d_G, d_T},
                               d_flags, n_v, pred, d_frontier, n_v);
    }

    tree.mark_dirty();
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_aff_all_flags));
    CUDA_CHECK(cudaFree(d_frontier));
}

template void update_sssp<CSR>(CSR& graph, Tree& tree, Edge* d_ins, int n_ins, Edge* d_del, int n_del);