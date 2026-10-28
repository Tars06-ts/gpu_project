// Loader for the shared test graphs in data/test/ (see data/test/README.md).
//
// Header-only, plain C++17, no CUDA dependency, so it can be included from .cpp and .cu files.
// Gives you:
//   - the initial graph as an edge list and as CSR (copy the CSR arrays to the GPU yourself)
//   - the update batches
//   - the expected distances after each batch, and a function to compare against them
//
// Typical use:
//   TestCase tc = load_test_case("data/test", "tiny");
//   // run static SSSP on tc.graph -> compare with tc.expected[0]
//   // for k in 1..tc.batches.size(): apply tc.batches[k-1], update SSSP, compare with tc.expected[k]
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace testgraph {

constexpr float INF = std::numeric_limits<float>::infinity();

struct Edge {
    int u, v;
    float w;
};

enum class UpdateType : uint8_t { INSERT, DELETE, UPDATE };

struct EdgeUpdate {
    int u, v;
    float w;  // unused for DELETE
    UpdateType type;
};

using UpdateBatch = std::vector<EdgeUpdate>;

// Edge list form of the graph. Edges are sorted by (u, v), no duplicates, no self loops.
struct EdgeListGraph {
    int n = 0;
    int source = 0;
    std::vector<Edge> edges;
};

// CSR form. Outgoing edges of vertex u are col_idx[row_ptr[u] .. row_ptr[u+1]).
struct CSRGraph {
    int n = 0;
    int m = 0;
    std::vector<int> row_ptr;   // size n + 1
    std::vector<int> col_idx;   // size m
    std::vector<float> weights; // size m
};

struct TestCase {
    std::string name;
    EdgeListGraph graph;                    // initial graph
    std::vector<UpdateBatch> batches;       // batches[k-1] is batch k
    std::vector<std::vector<float>> expected;  // expected[k] = distances after batches 1..k (expected[0] = initial)
};

namespace detail {

// Reads the next non-empty, non-comment line.
inline bool next_line(std::istream& in, std::istringstream& out) {
    std::string line;
    while (std::getline(in, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        out.clear();
        out.str(line);
        return true;
    }
    return false;
}

inline std::ifstream open(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    return f;
}

inline bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

}  // namespace detail

inline EdgeListGraph read_graph(const std::string& path) {
    auto f = detail::open(path);
    std::istringstream ls;
    EdgeListGraph g;
    int m;
    if (!detail::next_line(f, ls) || !(ls >> g.n >> m >> g.source))
        throw std::runtime_error(path + ": bad header");
    g.edges.reserve(m);
    for (int i = 0; i < m; i++) {
        Edge e;
        if (!detail::next_line(f, ls) || !(ls >> e.u >> e.v >> e.w))
            throw std::runtime_error(path + ": bad edge line " + std::to_string(i));
        g.edges.push_back(e);
    }
    return g;
}

inline UpdateBatch read_batch(const std::string& path) {
    auto f = detail::open(path);
    std::istringstream ls;
    int k;
    if (!detail::next_line(f, ls) || !(ls >> k))
        throw std::runtime_error(path + ": bad header");
    UpdateBatch b;
    b.reserve(k);
    for (int i = 0; i < k; i++) {
        char op;
        EdgeUpdate up{0, 0, 0.0f, UpdateType::INSERT};
        if (!detail::next_line(f, ls) || !(ls >> op >> up.u >> up.v))
            throw std::runtime_error(path + ": bad update line " + std::to_string(i));
        switch (op) {
            case 'I': up.type = UpdateType::INSERT; ls >> up.w; break;
            case 'D': up.type = UpdateType::DELETE; break;
            case 'U': up.type = UpdateType::UPDATE; ls >> up.w; break;
            default: throw std::runtime_error(path + ": unknown op " + std::string(1, op));
        }
        b.push_back(up);
    }
    return b;
}

inline std::vector<float> read_expected(const std::string& path) {
    auto f = detail::open(path);
    std::istringstream ls;
    int n;
    if (!detail::next_line(f, ls) || !(ls >> n))
        throw std::runtime_error(path + ": bad header");
    std::vector<float> dist(n, INF);
    for (int i = 0; i < n; i++) {
        int v;
        std::string d;
        if (!detail::next_line(f, ls) || !(ls >> v >> d))
            throw std::runtime_error(path + ": bad line " + std::to_string(i));
        dist[v] = (d == "inf") ? INF : std::stof(d);
    }
    return dist;
}

// Loads <dir>/<name>.graph, all <name>.batchK and all <name>.expectedK that exist.
inline TestCase load_test_case(const std::string& dir, const std::string& name) {
    TestCase tc;
    tc.name = name;
    std::string base = dir + "/" + name;
    tc.graph = read_graph(base + ".graph");
    tc.expected.push_back(read_expected(base + ".expected0"));
    for (int k = 1; detail::file_exists(base + ".batch" + std::to_string(k)); k++) {
        tc.batches.push_back(read_batch(base + ".batch" + std::to_string(k)));
        tc.expected.push_back(read_expected(base + ".expected" + std::to_string(k)));
    }
    return tc;
}

inline CSRGraph to_csr(const EdgeListGraph& g) {
    CSRGraph c;
    c.n = g.n;
    c.m = static_cast<int>(g.edges.size());
    c.row_ptr.assign(g.n + 1, 0);
    for (const Edge& e : g.edges) c.row_ptr[e.u + 1]++;
    for (int i = 0; i < g.n; i++) c.row_ptr[i + 1] += c.row_ptr[i];
    c.col_idx.resize(c.m);
    c.weights.resize(c.m);
    std::vector<int> pos(c.row_ptr.begin(), c.row_ptr.end() - 1);
    for (const Edge& e : g.edges) {
        int p = pos[e.u]++;
        c.col_idx[p] = e.v;
        c.weights[p] = e.w;
    }
    return c;
}

// Simple CPU reference for applying a batch to an edge list, in order.
// Slow (rebuilds a map); meant for tests only, not for the real graph API.
inline void apply_batch(EdgeListGraph& g, const UpdateBatch& batch) {
    std::map<std::pair<int, int>, float> em;
    for (const Edge& e : g.edges) em[{e.u, e.v}] = e.w;
    for (const EdgeUpdate& up : batch) {
        auto key = std::make_pair(up.u, up.v);
        switch (up.type) {
            case UpdateType::INSERT: em[key] = up.w; break;
            case UpdateType::DELETE: em.erase(key); break;
            case UpdateType::UPDATE: em[key] = up.w; break;
        }
    }
    g.edges.clear();
    for (const auto& [k, w] : em) g.edges.push_back({k.first, k.second, w});
}

// Compares distances exactly (weights are integers, so float sums are exact).
// Prints up to max_report mismatches. Returns true if everything matches.
inline bool check_distances(const std::vector<float>& got, const std::vector<float>& expected,
                            const std::string& label = "", int max_report = 10) {
    if (got.size() != expected.size()) {
        std::cerr << "[FAIL] " << label << ": size " << got.size() << " != " << expected.size() << "\n";
        return false;
    }
    int bad = 0;
    for (size_t v = 0; v < got.size(); v++) {
        bool same = (std::isinf(got[v]) && std::isinf(expected[v])) || got[v] == expected[v];
        if (!same) {
            if (bad < max_report)
                std::cerr << "  vertex " << v << ": got " << got[v] << ", expected " << expected[v] << "\n";
            bad++;
        }
    }
    if (bad) std::cerr << "[FAIL] " << label << ": " << bad << " / " << got.size() << " vertices wrong\n";
    else std::cerr << "[ OK ] " << label << "\n";
    return bad == 0;
}

}  // namespace testgraph
