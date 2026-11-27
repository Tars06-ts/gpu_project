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
};

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
	int vmfb(O* op, int n_op, F&& f, bool* flags, int n_f, P&& pred, int* buf, int n_buf);
};

#endif
