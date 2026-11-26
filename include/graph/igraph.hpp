#ifndef IGRAPH_HPP
#define IGRAPH_HPP

#include <vector>

struct Edge{
    int u, v, weight;
};

struct Neighbor{
    int v, weight;
};

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

class CSR : public IGraph {
	void expand();
	void contract();
	// number of vertices, edges
	int n_v, n_e;
	// track valid and invalid (removed) vertices
	std::vector<bool> valid;
	// edges and weights stored as in- and out-neighbors
	// trading data duplication with faster access
	std::vector<int> in_offset, in_edges, in_weights;
	std::vector<int> out_offset, out_edges, out_weights;

public:
	/* initializes graph with given number of vertices
	 * n_v: number of vertices*/
	CSR(int n_v);
	/* builds graphs from edge list
	 * n_v: number of vertices
	 * n_e: number of edges
	 * edge_list: (u,v,weight) triplets*/
	CSR(int n_v, Edge* edge_list, int n_e);
	~CSR() = default;

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
	/* returns weight of edge from u to v
	 * -1 if invalid (assume positive weights only)*/
	int get_weight(int u, int v) override;
	/* fills into buf atmost n in-neighbors of u
	 * returns number of in-neighbors filled into buf*/
	int get_in_neighbors(int u, Neighbor* buf, int n) override;
	/* fills into buf atmost n out-neighbors of u
	 * returns number of out-neighbors filled into buf*/
	int get_out_neighbors(int u, Neighbor* buf, int n) override;
	/* fills into buf atmost n edges of Graph
	 * returns number of edges filled into buf*/
	int edge_list(Edge* buf, int n) override;
	/* removes all edges of vertex and marks vertex to be deleted*/
	void remove_vertex(int u) override;
	/* applies n_ins insertions and n_del deletions
	 * ignores invalid deletions of non-existent vertices or edges
	 * updates weights for insertion of existing edges
	 * inserts new vertices if required by insertions*/
	void apply(Edge* ins, int n_ins, 
					   Edge* del, int n_del) override;
};

#endif
