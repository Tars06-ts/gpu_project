#ifndef SLAB_CUH
#define SLAB_CUH

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "graph/igraph.cuh"
#include "graph/types.cuh"

/* ---------------------------------------------------------------------
 * SLAB LAYOUT  (one slab = 32 x 4B = 128B = one warp-wide coalesced load)
 *
 *   word : 0  1 | 2  3 | 4  5 | ...  | 28 29 | 30     | 31
 *   use  : k0 w0| k1 w1| k2 w2| ...  | k14 w14| unused | next slab idx
 *
 *  - 15 (key, weight) pairs. key = neighbour vertex id, weight = edge weight
 *  - key words sit on EVEN lanes 0..28, so lane l owns pair l/2 when l is even
 *  - a pair is 8B aligned, so key+weight can be claimed with ONE 64-bit CAS
 *  - word 31 = index of next slab in the pool (NULL_SLAB = end of list)
 * ------------------------------------------------------------------- */
constexpr uint32_t SLAB_WORDS   = 32;
constexpr uint32_t PAIRS_PER    = 15;
constexpr uint32_t NEXT_LANE    = 31;
constexpr uint32_t EMPTY_KEY    = 0xFFFFFFFFu;   // never used slot (pool is memset to 0xFF)
constexpr uint32_t TOMBSTONE    = 0xFFFFFFFEu;   // deleted slot, reusable by inserts
constexpr uint32_t NULL_SLAB    = 0xFFFFFFFFu;
constexpr uint32_t KEY_LANES    = 0x15555555u;   // bits 0,2,4,...,28
constexpr uint32_t FULL_MASK    = 0xFFFFFFFFu;

// one direction (out-edges OR in-edges) of the graph
struct SlabDir {
	uint32_t* pool;        // pool_cap slabs. slab v (v < max_n_v) is the HEAD slab of vertex v
	uint32_t* next_free;   // device scalar: bump allocator, next unused slab index
	uint32_t  pool_cap;    // total slabs in pool
	int*      deg;         // live degree per vertex
};

// device-side view, same role as CSRView
struct SlabView {
	int max_n_v;
	// keep these four contiguous (push/pull_scalars copy them in one go)
	int n_v, n_e, max_v, min_v;
	bool* valid;
	SlabDir out;
	SlabDir in;
	int n_failed;
	int* failed_idx;
};

class SlabGraph : public IGraph {
public:
	// extra_slabs = slabs available beyond the max_n_v head slabs (per direction).
	// -1 -> pick a default from n_e
	SlabGraph(int n_v, int max_n_v, Edge* edge_list, int n_e, int extra_slabs = -1);
	~SlabGraph() override;

	// ---- IGraph ----
	int  num_vertices() override;
	int  num_edges() override;
	int  get_in_deg(int v) override;
	int  get_out_deg(int v) override;
	bool has_edge(int u, int v) override;
	int  is_valid(int u) override;
	int  get_weight(int u, int v) override;
	int  get_in_neighbors(int u, Neighbor* buf, int n) override;
	int  get_out_neighbors(int u, Neighbor* buf, int n) override;
	int  edge_list(Edge* buf, int n) override;
	void remove_vertex(int u) override;
	void apply(Edge* ins, int n_ins, Edge* del, int n_del) override;

	// ---- device management (same contract as CSR) ----
	SlabView* to_gpu();
	SlabView* view();
	void      free_gpu();

private:
	// host-side scalars only; the graph itself lives on the device
	int n_v, n_e, max_n_v, max_v, min_v;
	int extra_slabs;
	std::vector<Edge> init_edges;      // consumed by to_gpu()

	SlabView* d_ptr = 0;
	bool*     d_valid = 0;
	SlabDir   h_out{}, h_in{};         // host copies of the device pointers (for cudaFree)

	void alloc_dir(SlabDir& d, uint32_t cap);
	void free_dir(SlabDir& d);
	void push_scalars();
	void pull_scalars();
};

#endif
