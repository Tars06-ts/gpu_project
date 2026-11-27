#ifndef G_LOADER_H
#define G_LOADER_H

#include <string>
#include <vector>
#include "common/types.h"

struct LoadOptions {
    bool one_indexed    = false; // shift 1-based vertex IDs to 0-based
    int  default_weight = 1;     // weight assigned when omitted in input
    int  max_vertices   = 0;     // 0 = infer from header/data; otherwise fixed capacity
};

// Loads an initial graph from file into GraphData.
//
// Supported formats:
//   1. Adjacency list:
//        <num_vertices>, <max_vertices>
//        <u_id>,<deg> <v1>,<w1> <v2>,<w2> ...
//
//   2. Edge list / DIMACS:
//        c / # / % ...          comment lines (ignored)
//        p [sp] <n_v> [extra]   optional header (n_v = vertex count)
//        [a] <u> <v> [w]      space- or comma-separated edge (w defaults to opt.default_weight)
GraphData load_graph(const std::string& file, const LoadOptions& opt = LoadOptions());

// Loads dynamic graph update batches from file.
// Blank lines or "---" separate consecutive batches.
//
// Operations:
//   I <u> <v> [w]   insert or update edge (u, v)
//   D <u> <v>       delete edge (u, v)
//   R <u>           remove vertex u
void load_batch(const std::string& file, std::vector<HostBatch>& out,
                const LoadOptions& opt = LoadOptions());

#endif // G_LOADER_H
