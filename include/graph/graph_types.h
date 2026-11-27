#ifndef GRAPH_TYPES_H
#define GRAPH_TYPES_H

#include <memory>
#include "graph/igraph.h"
#include "common/types.h"

// different types of graphs
enum class GraphKind {
	CSR_CPU,
	CSR_GPU,
	SLAB_GPU,
};

// loader for graph
std::unique_ptr<IGraph> make_graph(GraphKind kind, const GraphData& d);

#endif
