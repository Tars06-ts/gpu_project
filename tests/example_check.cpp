// Example of how to use include/common/test_graph.h.
// Runs a simple CPU Dijkstra (recompute after every batch) on every test graph and checks
// it against the expected files. Replace `sssp` with your own CPU/GPU implementation.
//
//   g++ -std=c++17 -O2 -I include tests/example_check.cpp -o example_check
//   ./example_check data/test
//
// With CUDA the same header works from a .cu file:
//   nvcc -std=c++17 -O2 -I include my_test.cu -o my_test
#include "common/test_graph.h"

#include <functional>
#include <queue>

using namespace testgraph;

static std::vector<float> sssp(const CSRGraph& g, int src) {
    std::vector<float> dist(g.n, INF);
    using Item = std::pair<float, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
    dist[src] = 0;
    pq.push({0, src});
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue;
        for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
            int v = g.col_idx[i];
            float nd = d + g.weights[i];
            if (nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }
    return dist;
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "data/test";
    bool all_ok = true;
    for (const char* name : {"tiny", "random1k"}) {
        TestCase tc = load_test_case(dir, name);
        EdgeListGraph g = tc.graph;
        all_ok &= check_distances(sssp(to_csr(g), g.source), tc.expected[0], tc.name + " initial");
        for (size_t k = 0; k < tc.batches.size(); k++) {
            apply_batch(g, tc.batches[k]);
            all_ok &= check_distances(sssp(to_csr(g), g.source), tc.expected[k + 1],
                                      tc.name + " after batch " + std::to_string(k + 1));
        }
    }
    return all_ok ? 0 : 1;
}
