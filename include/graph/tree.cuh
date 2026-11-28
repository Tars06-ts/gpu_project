#ifndef TREE_H
#define TREE_H

#include <vector>
#include <climits>
#include <cuda_runtime.h>
#include "graph/checks.cuh"

static constexpr int INF_DIST = INT_MAX;

struct TreeView {
    int max_n_v;
    int source;
    int* dist;
    int* parent;
    bool* exist;
};

class Tree {
private:
    int max_n_v;
    int source;
    std::vector<int> distance;
    std::vector<int> parent;
    std::vector<bool> exist;

    // Device pointers
    int* d_dist;
    int* d_parent;
    bool* d_exist;
    bool gpu_dirty;

public:
    Tree(int max_size, int src = 0)
        : max_n_v(max_size), source(src),
          distance(max_size, INF_DIST),
          parent(max_size, -1),
          exist(max_size, false),
          d_dist(nullptr), d_parent(nullptr), d_exist(nullptr),
          gpu_dirty(false) {
        if (src >= 0 && src < max_size) {
            distance[src] = 0;
            parent[src] = -1;
            exist[src] = true;
        }
    }

    ~Tree() {
        free_gpu();
    }

    void modify(int vertex, int new_parent, int new_distance) {
        if (vertex < 0 || vertex >= max_n_v) return;
        exist[vertex] = (new_distance != INF_DIST);
        parent[vertex] = new_parent;
        distance[vertex] = new_distance;
    }

    int dis(int u) {
        if (gpu_dirty) sync_to_host();
        return distance[u];
    }

    int get_parent(int u) {
        if (gpu_dirty) sync_to_host();
        return parent[u];
    }

    bool edge_exist(int u, int v) {
        if (gpu_dirty) sync_to_host();
        return (v >= 0 && v < max_n_v) && (parent[v] == u);
    }

    int size() const { return max_n_v; }
    int get_source() const { return source; }

    // Upload tree arrays to GPU and return a by-value TreeView for kernels
    TreeView to_gpu() {
        if (d_dist == nullptr) {
            CUDA_CHECK(cudaMalloc(&d_dist, sizeof(int) * max_n_v));
            CUDA_CHECK(cudaMalloc(&d_parent, sizeof(int) * max_n_v));
            CUDA_CHECK(cudaMalloc(&d_exist, sizeof(bool) * max_n_v));
        }
        CUDA_CHECK(cudaMemcpy(d_dist, distance.data(), sizeof(int) * max_n_v, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_parent, parent.data(), sizeof(int) * max_n_v, cudaMemcpyHostToDevice));

        std::vector<char> h_exist(exist.begin(), exist.end());
        CUDA_CHECK(cudaMemcpy(d_exist, h_exist.data(), sizeof(bool) * max_n_v, cudaMemcpyHostToDevice));

        gpu_dirty = false;
        return view();
    }

    TreeView view() const {
        return TreeView{max_n_v, source, d_dist, d_parent, d_exist};
    }

    void mark_dirty() { gpu_dirty = true; }

    void sync_to_host() {
        if (d_dist == nullptr) return;
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(distance.data(), d_dist, sizeof(int) * max_n_v, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(parent.data(), d_parent, sizeof(int) * max_n_v, cudaMemcpyDeviceToHost));

        std::vector<char> h_exist(max_n_v);
        CUDA_CHECK(cudaMemcpy(h_exist.data(), d_exist, sizeof(bool) * max_n_v, cudaMemcpyDeviceToHost));
        exist.assign(h_exist.begin(), h_exist.end());

        gpu_dirty = false;
    }

    void free_gpu() {
        if (d_dist != nullptr) { cudaFree(d_dist); d_dist = nullptr; }
        if (d_parent != nullptr) { cudaFree(d_parent); d_parent = nullptr; }
        if (d_exist != nullptr) { cudaFree(d_exist); d_exist = nullptr; }
        gpu_dirty = false;
    }
};

#endif