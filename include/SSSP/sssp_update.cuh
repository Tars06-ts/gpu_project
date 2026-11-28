#ifndef SSSP_UPDATE_H
#define SSSP_UPDATE_H

#include "graph/tree.cuh"
#include "graph/types.cuh"

template<typename GraphImpl>
void update_sssp(GraphImpl& graph, Tree& tree, Edge* d_ins, int n_ins, Edge* d_del, int n_del);

#endif