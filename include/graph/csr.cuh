#ifndef CSR_H
#define CSR_H

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "graph/igraph.h"
#include "graph/types.cuh"
#include <algorithm>
#include "graph/checks.cuh"

struct CSRView {
	// maximum number of vertices
	int max_n_v;
	// number of active vertices, edges
	int n_v, n_e;
	// index of max, min active vertex
	int max_v, min_v;
	// track valid and invalid (removed) vertices
	bool* valid;
	// edges and weights stored as in- and out-neighbors
	// trading data duplication with faster access
	// *_offset has n_v+1 elements
	// *_edges, *_weights has n_e elements
	int *in_offset, *in_edges, *in_weights, *in_used;
	int *out_offset, *out_edges, *out_weights, *out_used;


	template<typename Fn>
	__device__ void for_each_out_neighbor(int u, Fn&& fn) const {
		int start = out_offset[u];
		int end = start + out_used[u];
		for (int i = start; i < end; ++i) {
			int v = out_edges[i];
			if (v >= 0) {
				fn(v, out_weights[i]);
			}
		}
	}

	template<typename Fn>
	__device__ void for_each_in_neighbor(int u, Fn&& fn) const {
		int start = in_offset[u];
		int end = start + in_used[u];
		for (int i = start; i < end; ++i) {
			int v = in_edges[i];
			if (v >= 0) {
				fn(v, in_weights[i]);
			}
		}
	}
	
};


// NOTE: As they are template function they need to be in header file 

template<class O, class F>
__global__ void d_helper_vmfb_mark(O* op, int n_op, F f, bool* flags, int n_f) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n_op) {
        f(op, n_op, flags, n_f, tid);
    }
}

template<class P>
__global__ void d_helper_vmfb_filter(const bool* flags, int n_f, P pred, int* buf, int n_buf, int* d_buf_count) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int lane_id = threadIdx.x & 31; // threadIdx.x % 32

    bool is_active = (tid < n_f) && pred(flags[tid]);
    unsigned int active_threads = __ballot_sync(0xFFFFFFFF, is_active);

    if (active_threads == 0) return;

    int leader_lane = __ffs(active_threads) - 1;
    int warp_offset = 0;

    if (lane_id == leader_lane) {
        int warp_total_active = __popc(active_threads);
        warp_offset = atomicAdd(d_buf_count, warp_total_active);
    }

    // Broadcast the base offset from leader_lane to all lanes in the warp
    warp_offset = __shfl_sync(0xFFFFFFFF, warp_offset, leader_lane);

    if (is_active) {
        int offset = __popc(((1u << lane_id) - 1) & active_threads);
        int idx = warp_offset + offset;
        if (idx < n_buf) {
            buf[idx] = tid;
        }
    }
}

// gives the kernel config (blocks, tpb)
static void launch_dims(long long threads, int& blocks, int& tpb){
	if (threads < 1) threads = 1;
	long long t = ((threads + 32 - 1) / 32) * 32;
	tpb = (int)(t < 1024 ? t : 1024); 
	long long b = (t + tpb - 1) / tpb;
	blocks = (int)(b < 65535 ? b : 65535);
}

class CSR : public IGraph {
	// maximum number of vertices
	int max_n_v;
	// number of active vertices, edges
	int n_v, n_e;
	// index of max, min active vertex
	int max_v, min_v;
	// track valid and invalid (removed) vertices
	std::vector<bool> valid;
	// edges and weights stored as in- and out-neighbors
	// trading data duplication with faster access
	// *_used indicates number of live edges per vertex
	// invalid edges have in_edges[*] = -1
	std::vector<int> in_offset, in_edges, in_weights, in_used;
	std::vector<int> out_offset, out_edges, out_weights, out_used;
	// for transferring into gpu
	int *d_n_v, *d_n_e;
	int *d_in_offset, *d_in_edges, *d_in_weights, *d_in_used;
	int *d_out_offset, *d_out_edges, *d_out_weights, *d_out_used;
	bool *d_valid;
	// pointer to graph inside gpu
	CSRView* d_ptr;
	bool gpu_dirty;

	// copies max_n_v, n_v, n_e, max_v, min_v from host to device
	void push_scalars();
	// copies max_n_v, n_v, n_e, max_v, min_v from device to host
	void pull_scalars();
	// copies pointer to new d_ptr field from host to device
	void push_ptr(size_t, int*);
	// helpers for apply
	void apply_deletions(Edge* del, int n_del, 
				cudaStream_t s1, cudaStream_t s2);
	void apply_insertions(Edge* ins, int n_ins, 
				cudaStream_t s1, cudaStream_t s2);
	// filters the insertion list of invalid, identical edges
	void ins_filter(InsertCtx& c, cudaStream_t s);
	// returns true if some vertex needs more space than it has
	bool ins_count_demand(InsertCtx& c, cudaStream_t s);
	// rebuilds the edge, weight, offsets to allow insertions
	void ins_rebuild(InsertCtx& c, cudaStream_t s1, cudaStream_t s2);
	// updates existing edges only
	void ins_update_existing(InsertCtx& c, cudaStream_t s1, cudaStream_t s2);
	// writes new vertices
	void ins_write_new(InsertCtx& c, cudaStream_t s1, cudaStream_t s2);
	// packs, i.e. moves active edges to beginning of array
	void pack_all(cudaStream_t s1, cudaStream_t s2);
public:
	/* initializes graph with given number of vertices
	 * n_v: number of vertices
	 * max_n_v: maximum number of vertices in graph
	 *			including ones that can be inserted in future*/
	CSR(int n_v, int max_n_v);
	/* builds graphs from edge list
	 * n_v: number of vertices
	 * max_n_v: maximum number of vertices in graph
	 *			including ones that can be inserted in future
	 * n_e: number of edges
	 * edge_list: (u,v,weight) triplets*/
	CSR(int n_v, int max_n_v, Edge* edge_list, int n_e);
	/* override destructor */
	~CSR() override;

    /* returns number of vertices in graph*/
	int num_vertices() override;
    /* returns number of edges in graph*/
    int num_edges() override;
    /* returns in degree of vertex*/
	int get_in_deg(int v) override;
    /* returns out degree of vertex*/
	int get_out_deg(int v) override;
	/* returns true if there is an edge from u to v, 
	 * false otherwise*/
	bool has_edge(int u, int v) override;
	/* returns true if valid vertex else false*/
	int is_valid(int u) override;
	/* returns weight of edge from u to v
	 * -1 if invalid (assume positive weights only)*/
	int get_weight(int u, int v) override;
	/* fills into buf atmost n in-neighbors of u
	 * assumes buffer has been allocated in device
	 * returns number of in-neighbors filled into buf*/
	int get_in_neighbors(int u, Neighbor* buf, int n) override;
	/* fills into buf atmost n out-neighbors of u
	 * assumes buffer has been allocated in device
	 * returns number of out-neighbors filled into buf*/
	int get_out_neighbors(int u, Neighbor* buf, int n) override;
	/* fills into buf atmost n edges of Graph
	 * assumes buffer has been allocated in device
	 * returns number of edges filled into buf*/
	int edge_list(Edge* buf, int n) override;
	/* removes all edges of vertex and marks vertex to be deleted*/
	void remove_vertex(int u) override;
	/* applies n_ins insertions and n_del deletions
	 * ignores invalid deletions of non-existent vertices or edges
	 * updates weights for insertion of existing edges
	 * inserts new vertices if required by insertions
	 * assumes ins, del have been allocated in device*/
	void apply(Edge* ins, int n_ins, 
			   Edge* del, int n_del) override;
	/* copies the graph to gpu as CSRView struct and returns pointer
	 * ignores if cpu's copy is stale compared to gpu's copy*/
	CSRView* to_gpu();
	/* synchronizes the graph between gpu to cpu */
	void sync_graph();
	/* returns device pointer to allocated graph in gpu
	 * undefined if to_gpu() is not called*/
	CSRView* view();
	/* free's the allocated memory*/
	void free_gpu();

	/* applies function f on n_op operands in op and
	 * modifies flags of length n_f and
	 * fills the list of vertices for whom predicate is true in buf and
	 * returns the number of vertices filled in buf of max size n_buf*/
	template<class O, class F, class P>
	int vmfb(O* op, int n_op, F f, bool* flags, int n_f,
				P pred, int* buf, int n_buf) {
		int blocks, tpb;

		// Vertex Marking
		if (n_op > 0) {
			launch_dims(n_op, blocks, tpb);
			d_helper_vmfb_mark<<<blocks, tpb>>>(op, n_op, f, flags, n_f);
			CUDA_CHECK(cudaGetLastError());
		}

		// Synchronization
		CUDA_CHECK(cudaDeviceSynchronize());

		// Filter
		if (n_f <= 0 || n_buf <= 0) return 0;

		int* d_buf_count = nullptr;
		CUDA_CHECK(cudaMalloc(&d_buf_count, sizeof(int)));
		CUDA_CHECK(cudaMemset(d_buf_count, 0, sizeof(int)));

		launch_dims(n_f, blocks, tpb);
		d_helper_vmfb_filter<<<blocks, tpb>>>(flags, n_f, pred, buf, n_buf, d_buf_count);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		int buf_actual_size = 0;
		CUDA_CHECK(cudaMemcpy(&buf_actual_size, d_buf_count, sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaFree(d_buf_count));

		return std::min(buf_actual_size, n_buf);
	}

};


#endif
