#ifndef IGRAPH_CUH
#define IGRAPH_CUH

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "graph/types.cuh"

class IGraph{
public:
	virtual ~IGraph() = default;
    /* returns number of vertices in graph*/
	virtual int num_vertices() = 0;
    /* returns number of edges in graph*/
    virtual int num_edges() = 0;
    /* returns in degree of vertex*/
	virtual int get_in_deg(int v) = 0;
    /* returns out degree of vertex*/
	virtual int get_out_deg(int v) = 0;
	/* returns true if there is an edge from u to v, 
	 * false otherwise*/
	virtual bool has_edge(int u, int v) = 0;
	/* returns true if valid vertex else false*/
	virtual int is_valid(int u) = 0;
	/* returns weight of edge from u to v
	 * -1 if invalid (assume positive weights only)*/
	virtual int get_weight(int u, int v) = 0;
	/* fills into buf atmost n in-neighbors of u
	 * returns number of in-neighbors filled into buf*/
	virtual int get_in_neighbors(int u, Neighbor* buf, int n) = 0;
	/* fills into buf atmost n out-neighbors of u
	 * returns number of out-neighbors filled into buf*/
	virtual int get_out_neighbors(int u, Neighbor* buf, int n) = 0;
	/* fills into buf atmost n edges of Graph
	 * returns number of edges filled into buf*/
	virtual int edge_list(Edge* buf, int n) = 0;
	/* removes all edges of vertex and marks vertex to be deleted*/
	virtual void remove_vertex(int u) = 0;
	/* applies n_ins insertions and n_del deletions
	 * ignores invalid deletions of non-existent vertices or edges
	 * updates weights for insertion of existing edges
	 * inserts new vertices if required by insertions*/
	virtual void apply(Edge* ins, int n_ins, 
					   Edge* del, int n_del) = 0;
};

#endif

