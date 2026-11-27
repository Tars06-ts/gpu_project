#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <vector>
#include "graph/types.cuh"

// Data required for initializing graph
struct GraphData {
	int n_v;
	int max_n_v;
	std::vector<Edge> edges;      // (u, v, weight), one entry per directed edge
};

// Batch has all inserts and updates together
struct Batch{
	// number of insertions, deletions, vertex removals
	int n_ins, n_del, n_rem;
	// insertions
	Edge* ins;
	// deletions
	Edge* del;
	// vertex removals
	int* v;
};

// Host's copy of Batch
struct HostBatch {
	std::vector<Edge> ins;    // from "I u v wt" lines
	std::vector<Edge> del;    // from "D u v" lines (weight field unused)
	std::vector<int>  rem;    // from "R u" lines
};

// SSSPResult has the resulting shortest path tree + 
//			  shortest distance from vertex 0
struct SSSPResult {
	// number of vertices
	int n;
	// distance from vertex 0
	int *dist;
	// parent of each vertex
	// = -1 if no parent
	int *parent;
};

#endif
