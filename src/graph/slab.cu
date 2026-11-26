#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <device_atomic_functions.h>
#include <driver_types.h>
#include <sys/types.h>
#include "graph/slab.cuh"
#include "graph/checks.cuh"
#include "graph/primitives.cuh"
#include "graph/types.cuh"
#include "graph/util.h"


static void launch_dims(long long threads, int& blocks, int& tpb){
	if (threads < 1) threads = 1;
	long long t = ((threads + 31) / 32) * 32;
	tpb = (int)(t < 1024 ? t : 1024);
	long long b = (t + tpb - 1) / tpb;
	blocks = (int)(b < 65535 ? b : 65535);
}

/* =====================================================================
 *  DEVICE: warp-cooperative slab primitives
 *
 *  Rule: ALL 32 lanes of a warp call these together with the SAME
 *  arguments (work-sharing: one warp = one operation). So the loops
 *  below are warp-uniform and the __*_sync calls are safe.
 * =================================================================== */

__device__ __forceinline__ int  lane_id()      { return threadIdx.x & 31; }
__device__ __forceinline__ int  warp_gid()     { return (blockIdx.x * blockDim.x + threadIdx.x) >> 5; }
__device__ __forceinline__ int  warps_total()  { return (gridDim.x * blockDim.x) >> 5; }
__device__ __forceinline__ bool is_key_lane(int lane){ return (KEY_LANES >> lane) & 1u; }

__device__ __forceinline__ uint32_t* slab_at(const SlabDir& d, uint32_t s){
	return d.pool + (size_t)s * SLAB_WORDS;
}

// grab a fresh slab from the pool (pool is pre-filled with 0xFF, so it is already empty)
// returns NULL_SLAB when the pool is exhausted
__device__ uint32_t alloc_slab(const SlabDir& d){
	uint32_t s = atomicAdd(d.next_free, 1u);
	return (s < d.pool_cap) ? s : NULL_SLAB;
}

// head slab of the bucket where `key` lives in u's list.
// u owns B_u = bucket_off[u+1]-bucket_off[u] buckets (always >= 1); key goes to bucket hash(key) % B_u
// TODO: use the hash function your course specifies (this is just a simple multiplicative hash)
__device__ __forceinline__ uint32_t head_of(const SlabDir& d, uint32_t u, uint32_t key){
	uint32_t B = d.bucket_off[u + 1] - d.bucket_off[u];
	return d.bucket_off[u] + (key * 2654435761u) % B;
}

// does edge (u -> key) exist in u's slab list?  weight returned in *val (all lanes)
__device__ bool warp_search(const SlabDir& d, uint32_t u, uint32_t key, uint32_t* val){
	const int lane = lane_id();
	uint32_t cur = head_of(d, u, key);                    // only this key's bucket needs searching
	while (cur != NULL_SLAB){
		const uint32_t* s = slab_at(d, cur);
		uint32_t w  = __ldcg(s + lane);                    // whole slab in one transaction
		uint32_t wn = __shfl_down_sync(FULL_MASK, w, 1);   // lane l <- word l+1 (the weight)
		uint32_t hit = __ballot_sync(FULL_MASK, is_key_lane(lane) && w == key);
		if (hit){
			int src = __ffs(hit) - 1;
			*val = __shfl_sync(FULL_MASK, wn, src);
			return true;
		}
		cur = __shfl_sync(FULL_MASK, w, NEXT_LANE);        // follow the next pointer
	}
	return false;
}

// insert (u -> key, val); if it already exists just overwrite the weight
//  returns  1: new edge inserted,  0: existed, weight updated,  -1: pool exhausted
__device__ int warp_upsert(const SlabDir& d, uint32_t u, uint32_t key, uint32_t val){
	const int lane = lane_id();

	// phase 1: the key can only be in its own bucket, but may be in ANY slab of that bucket's chain
	for (uint32_t cur = head_of(d, u, key); cur != NULL_SLAB; ){
		uint32_t* s = slab_at(d, cur);
		uint32_t w = __ldcg(s + lane);
		uint32_t hit = __ballot_sync(FULL_MASK, is_key_lane(lane) && w == key);
		if (hit){
			int src = __ffs(hit) - 1;
			if (lane == src) atomicExch(s + lane + 1, val);
			return 0;
		}
		cur = __shfl_sync(FULL_MASK, w, NEXT_LANE);
	}

	// phase 2: claim the first EMPTY or TOMBSTONE slot
	uint32_t cur = head_of(d, u, key);
	while (true){
		uint32_t* s = slab_at(d, cur);
		uint32_t w  = __ldcg(s + lane);
		uint32_t wn = __shfl_down_sync(FULL_MASK, w, 1);
		uint32_t free_m = __ballot_sync(FULL_MASK,
		                      is_key_lane(lane) && (w == EMPTY_KEY || w == TOMBSTONE));
		if (free_m){
			int win = __ffs(free_m) - 1;                   // elected lane
			int ok = 0;
			if (lane == win){
				// claim key AND weight with a single 64-bit CAS
				unsigned long long expect = ((unsigned long long)wn << 32) | w;
				unsigned long long want   = ((unsigned long long)val << 32) | key;
				ok = (atomicCAS((unsigned long long*)(s + lane), expect, want) == expect);
			}
			ok = __shfl_sync(FULL_MASK, ok, win);
			if (ok) return 1;
			continue;                                      // lost the race: re-read same slab
		}
		// slab is full
		uint32_t nxt = __shfl_sync(FULL_MASK, w, NEXT_LANE);
		if (nxt == NULL_SLAB){
			uint32_t fresh = NULL_SLAB;
			if (lane == 0) fresh = alloc_slab(d);
			fresh = __shfl_sync(FULL_MASK, fresh, 0);
			if (fresh == NULL_SLAB) return -1;
			if (lane == 0) atomicCAS(s + NEXT_LANE, NULL_SLAB, fresh);  // if we lose, our slab is wasted
			continue;                                      // re-read: next ptr now set (by us or winner)
		}
		cur = nxt;
	}
}

// delete (u -> key): turn the slot into a tombstone. returns 1 if this call removed it
__device__ int warp_delete(const SlabDir& d, uint32_t u, uint32_t key){
	const int lane = lane_id();
	for (uint32_t cur = head_of(d, u, key); cur != NULL_SLAB; ){
		uint32_t* s = slab_at(d, cur);
		uint32_t w = __ldcg(s + lane);
		uint32_t hit = __ballot_sync(FULL_MASK, is_key_lane(lane) && w == key);
		if (hit){
			int src = __ffs(hit) - 1;
			int ok = 0;
			if (lane == src) ok = (atomicCAS(s + lane, key, TOMBSTONE) == key);
			return __shfl_sync(FULL_MASK, ok, src);
		}
		cur = __shfl_sync(FULL_MASK, w, NEXT_LANE);
	}
	return 0;
}

/* =====================================================================
 *  KERNELS  (one warp per update)
 * =================================================================== */

// pool[i] == false -> skip (use it for the output of your drop-invalid / dedup step)

__global__ void k_insert(SlabView* G, const Edge* ins, int n_ins, const bool* pool){
	for (int i = warp_gid(); i < n_ins; i += warps_total()){
		if (pool && !pool[i]) continue;                    // warp-uniform
		Edge e = ins[i];
		int r_out = warp_upsert(G->out, e.u, e.v, e.weight);   // u's out-list gets v
		int r_in = warp_upsert(G->in,  e.v, e.u, e.weight);           // v's in-list gets u
		if (lane_id() == 0 ){ 
			if (r_out==1) {
				atomicAdd(&G->out.deg[e.u], 1); 
				atomicAdd(&G->n_e, 1);
			}     
			if (r_in == 1) atomicAdd(&G->in.deg[e.v], 1);
			
		}
		if (lane_id() == 0 && (r_in ==-1 || r_out == -1)){
			int slot= atomicAdd(&G->n_failed, 1);
			G->failed_idx[slot] = i;
		}
		// DONE: what should happen if r == -1 (pool exhausted)? at least don't lose it silently
	}
}

__global__ void k_delete(SlabView* G, const Edge* del, int n_del){
	for (int i = warp_gid(); i < n_del; i += warps_total()){
		Edge e = del[i];
		if (e.u < 0 || e.u >= G->max_n_v || e.v < 0 || e.v >= G->max_n_v) continue;
		int r_out = warp_delete(G->out, e.u, e.v);
		int r_in = warp_delete(G->in, e.v, e.u);
		if (lane_id() == 0){
			if (r_out == 1){
				atomicSub(&G->out.deg[e.u], 1);
				atomicSub(&G->n_e, 1);
			}
			if (r_in == 1)atomicSub(&G->in.deg[e.v], 1);
		}
	}
}

__global__ void k_neighbours(SlabDir d, int u, Neighbor* buf, int n_buf, int* count){
	const int lane = lane_id();
	// u now has several buckets: walk the chain of each one
	for (uint32_t b = d.bucket_off[u]; b < d.bucket_off[u + 1]; b++){
		uint32_t cur = b;
		while (cur != NULL_SLAB){
			const uint32_t* s = slab_at(d, cur);
			uint32_t key = __ldcg(s+lane);
			uint32_t wt = __shfl_down_sync(FULL_MASK, key, 1);

			bool live = is_key_lane(lane) && key !=EMPTY_KEY && key != TOMBSTONE;
			uint32_t bits = __ballot_sync(FULL_MASK, live);
			int base=0;
			if (lane == 0){
				base = atomicAdd(count, __popc(bits));
			}
			base = __shfl_sync(FULL_MASK, base, 0);
			if (live){
				int slot = base +__popc(bits & ((1u << lane) - 1));
				if (slot < n_buf) buf[slot] = {(int)key, (int)wt};
			}
			cur = __shfl_sync(FULL_MASK, key, NEXT_LANE);
		}
	}
}

__global__ void k_edge_list(SlabView* G, Edge* buf, int n_buf, int* count){
	const int lane = lane_id();
	for (int u = warp_gid(); u< G->max_n_v; u+=warps_total()){
		if (!G->valid[u]) continue;
		// walk every bucket of u's out-list
		for (uint32_t b = G->out.bucket_off[u]; b < G->out.bucket_off[u + 1]; b++){
			uint32_t cur = b;
			while (cur != NULL_SLAB){
				const uint32_t* s = slab_at(G->out, cur);
				uint32_t key = __ldcg(s+lane);
				uint32_t wt = __shfl_down_sync(FULL_MASK, key, 1);

				bool live = is_key_lane(lane) && key !=EMPTY_KEY && key != TOMBSTONE;
				uint32_t bits = __ballot_sync(FULL_MASK, live);
				int base=0;
				if (lane == 0){
					base = atomicAdd(count, __popc(bits));
				}
				base = __shfl_sync(FULL_MASK, base, 0);
				if (live){
					int slot = base +__popc(bits & ((1u << lane) - 1));
					if (slot < n_buf) buf[slot] = {u, (int)key, (int)wt};
				}
				cur = __shfl_sync(FULL_MASK, key, NEXT_LANE);
			}
		}
	}
}

__global__ void k_remove_vertex(SlabView* G, int u){
	const int lane = lane_id();
	for (int dir = 0; dir < 2; dir ++){
		const SlabDir& mine = (dir == 0) ? G->out : G->in;
		const SlabDir& other = (dir ==0) ? G->in : G->out;

		// walk every bucket of u's list
		for (uint32_t b = mine.bucket_off[u]; b < mine.bucket_off[u + 1]; b++){
			uint32_t cur = b;
			while (cur != NULL_SLAB){
				const uint32_t* s = slab_at(mine, cur);
				uint32_t key = __ldcg(s+lane);

				bool live = is_key_lane(lane) && key !=EMPTY_KEY && key != TOMBSTONE;
				uint32_t bits = __ballot_sync(FULL_MASK, live);
				int base=0;
				while (bits){
					int src = __ffs(bits) - 1;
					uint32_t nb = __shfl_sync(FULL_MASK, key, src);
					int r = warp_delete(other, nb, u);
					if (lane == 0 && r==1){
							atomicSub(&other.deg[nb], 1);
							atomicSub(&G->n_e, 1);
					}
					bits&=bits -1;
				}
				cur = __shfl_sync(FULL_MASK, key, NEXT_LANE);
			}
		}
	}
	// TODO: clear u's own lists (tombstone every live key in ALL of u's buckets, both directions),
	//       set out.deg[u] = in.deg[u] = 0, and finally valid[u] = false (lane 0)
}

__global__ void k_get_weights(SlabDir d, int u, int v, int *out){
	uint32_t val = 0;
	bool found = warp_search(d, u, v, &val);
	if (lane_id() == 0){
		*out = found ? (int)val : -1;
	}
}

// DONE k_neighbors(SlabDir d, int u, Neighbor* buf, int n_buf, int* count)
//   single warp walks u's chain. per slab:
//     live  = is_key_lane(lane) && key != EMPTY_KEY && key != TOMBSTONE
//     bits  = __ballot_sync(FULL_MASK, live)
//     lane 0: base = atomicAdd(count, __popc(bits)) ; broadcast base with __shfl_sync
//     each live lane writes to base + popc(bits & ((1u << lane) - 1)) if < n_buf
//     (weight is in the next lane -> __shfl_down_sync)
// DONE k_edge_list(...)           same pattern, warp per vertex u in [min_v, max_v]
// DONE k_remove_vertex(...)       see notes: you have in+out lists, so you don't need to scan the whole graph

/* =====================================================================
 *  HOST
 * =================================================================== */

SlabGraph::SlabGraph(int n_v, int max_n_v, Edge* edge_list, int n_e, int extra_slabs){
	this->n_v = n_v;
	this->max_n_v = max_n_v;
	this->max_v = n_v - 1;
	this->min_v = 0;
	this->n_e = 0;                         // becomes n_e once to_gpu() has inserted the edges
	if (extra_slabs < 0){
		extra_slabs = max_n_v + 2 * (n_e / (int)PAIRS_PER) + 1024;   // crude default, tune it
	}
	this->extra_slabs = extra_slabs;
	if (edge_list && n_e > 0) init_edges.assign(edge_list, edge_list + n_e);
}

SlabGraph::~SlabGraph(){ free_gpu(); }

// bucket_off has max_n_v+1 entries: vertex u owns head slabs [bucket_off[u], bucket_off[u+1])
// extra = dynamic slabs available on top of the head slabs
void SlabGraph::alloc_dir(SlabDir& d, const std::vector<uint32_t>& bucket_off, uint32_t extra){
	uint32_t base_slabs = bucket_off.back();                // total head slabs = sum of B_u
	uint32_t cap = base_slabs + extra;
	d.pool_cap = cap;
	CUDA_CHECK(cudaMalloc((void**)&d.pool, (size_t)cap * SLAB_WORDS * sizeof(uint32_t)));
	CUDA_CHECK(cudaMemset(d.pool, 0xFF, (size_t)cap * SLAB_WORDS * sizeof(uint32_t)));  // EMPTY keys, NULL next ptrs
	CUDA_CHECK(cudaMalloc((void**)&d.next_free, sizeof(uint32_t)));
	CUDA_CHECK(cudaMemcpy(d.next_free, &base_slabs, sizeof(uint32_t), cudaMemcpyHostToDevice));  // dynamic slabs start after the heads
	CUDA_CHECK(cudaMalloc((void**)&d.bucket_off, sizeof(uint32_t) * bucket_off.size()));
	CUDA_CHECK(cudaMemcpy(d.bucket_off, bucket_off.data(), sizeof(uint32_t) * bucket_off.size(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMalloc((void**)&d.deg, sizeof(int) * max_n_v));
	CUDA_CHECK(cudaMemset(d.deg, 0, sizeof(int) * max_n_v));
}

void SlabGraph::free_dir(SlabDir& d){
	cudaFree(d.pool); cudaFree(d.next_free); cudaFree(d.deg); cudaFree(d.bucket_off);
	d = SlabDir{};
}

SlabView* SlabGraph::to_gpu(){
	if (d_ptr != 0) return d_ptr;                          // already uploaded

	// bucket_off[u] = index of u's first head slab, B_u = bucket_off[u+1] - bucket_off[u]
	// TODO: count out-/in-degrees from init_edges on the host and choose
	//       B_u = max(1, ceil(deg / (load_factor * PAIRS_PER))) separately for out and in.
	//       Every vertex id (even invalid ones) needs B_u >= 1, they may be activated later.
	// for now: exactly 1 bucket per vertex (same behaviour as the single-list version)
	std::vector<uint32_t> out_off(max_n_v + 1), in_off(max_n_v + 1);
	for (int i = 0; i <= max_n_v; i++){ out_off[i] = i; in_off[i] = i; }
	alloc_dir(h_out, out_off, (uint32_t)extra_slabs);
	alloc_dir(h_in,  in_off,  (uint32_t)extra_slabs);

	std::vector<char> hv(max_n_v, 0);
	std::fill(hv.begin(), hv.begin() + n_v, 1);
	CUDA_CHECK(cudaMalloc((void**)&d_valid, max_n_v));
	CUDA_CHECK(cudaMemcpy(d_valid, hv.data(), max_n_v, cudaMemcpyHostToDevice));

	SlabView h{};
	h.max_n_v = max_n_v; h.n_v = n_v; h.n_e = 0; h.max_v = max_v; h.min_v = min_v;
	h.valid = d_valid; h.out = h_out; h.in = h_in;
	CUDA_CHECK(cudaMalloc((void**)&d_ptr, sizeof(SlabView)));
	CUDA_CHECK(cudaMemcpy(d_ptr, &h, sizeof(h), cudaMemcpyHostToDevice));

	// bulk-load the initial edges through the normal insert path
	if (!init_edges.empty()){
		Edge* d_e;
		size_t bytes = sizeof(Edge) * init_edges.size();
		CUDA_CHECK(cudaMalloc((void**)&d_e, bytes));
		CUDA_CHECK(cudaMemcpy(d_e, init_edges.data(), bytes, cudaMemcpyHostToDevice));
		int blocks, tpb;
		launch_dims((long long)init_edges.size() * 32, blocks, tpb);
		k_insert<<<blocks, tpb>>>(d_ptr, d_e, (int)init_edges.size(), nullptr);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		cudaFree(d_e);
		pull_scalars();
		init_edges.clear();
		init_edges.shrink_to_fit();
	}
	return d_ptr;
}

SlabView* SlabGraph::view(){ return d_ptr; }

void SlabGraph::free_gpu(){
	free_dir(h_out);
	free_dir(h_in);
	cudaFree(d_valid);
	cudaFree(d_ptr);
	d_valid = 0; d_ptr = 0;
}

void SlabGraph::push_scalars(){
	int h[4] = {n_v, n_e, max_v, min_v};
	CUDA_CHECK(cudaMemcpy((char*)d_ptr + offsetof(SlabView, n_v), h, sizeof(h), cudaMemcpyHostToDevice));
}
void SlabGraph::pull_scalars(){
	int h[4];
	CUDA_CHECK(cudaMemcpy(h, (char*)d_ptr + offsetof(SlabView, n_v), sizeof(h), cudaMemcpyDeviceToHost));
	n_v = h[0]; n_e = h[1]; max_v = h[2]; min_v = h[3];
}

void SlabGraph::apply(Edge* ins, int n_ins, Edge* del, int n_del){
	if (d_ptr == 0){
		FATAL("Invalid copy of Graph in device. to_gpu() hasn't been called yet, or it failed\n");
		return;
	}
	// NOTE: ins/del are device pointers here, same as in CSR::apply
	int blocks, tpb;

	if (n_del > 0){
		launch_dims((long long)n_del * 32, blocks, tpb);
		k_delete<<<blocks, tpb>>>(d_ptr, del, n_del);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (n_ins > 0){
		// TODO (same as CSR::ins_filter): drop out-of-range ids, dedup (keep latest per (u,v)),
		//      mark + activate new vertices, update n_v / min_v / max_v, then push_scalars().
		//      -> gives you a bool pool[n_ins] to pass instead of nullptr below.

		// buffer where the kernel records indices (into ins) of edges that failed to insert
		int* d_failed;
		CUDA_CHECK(cudaMalloc((void**)&d_failed, sizeof(int) * n_ins));
		CUDA_CHECK(cudaMemcpy((char*)d_ptr + offsetof(SlabView, failed_idx),
		                      &d_failed, sizeof(int*), cudaMemcpyHostToDevice));
		// reset the failure counter (it lives inside the device struct)
		CUDA_CHECK(cudaMemset((char*)d_ptr + offsetof(SlabView, n_failed), 0, sizeof(int)));

		launch_dims((long long)n_ins * 32, blocks, tpb);
		k_insert<<<blocks, tpb>>>(d_ptr, ins, n_ins, nullptr);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// how many insertions failed?
		int n_failed = 0;
		CUDA_CHECK(cudaMemcpy(&n_failed, (char*)d_ptr + offsetof(SlabView, n_failed),
		                      sizeof(int), cudaMemcpyDeviceToHost));
		if (n_failed > 0){
			// TODO: grow the pool, then re-run k_insert on just the edges
			//       ins[d_failed[0..n_failed)]  (d_failed holds indices, not edges)
		}

		cudaFree(d_failed);
	}
	pull_scalars();   // n_e was updated on the device by the kernels

	
	// TODO: check h_out.next_free / h_in.next_free against pool_cap -> pool exhausted?
}

/* ---- trivial getters: scalars live on device, pull on demand ---- */
int SlabGraph::num_vertices(){ if (d_ptr) pull_scalars(); return n_v; }
int SlabGraph::num_edges(){    if (d_ptr) pull_scalars(); return n_e; }

int SlabGraph::get_out_deg(int v){
	if (v < 0 || v >= max_n_v) return -1;
	int d = 0;
	CUDA_CHECK(cudaMemcpy(&d, h_out.deg + v, sizeof(int), cudaMemcpyDeviceToHost));
	return d;
}
int SlabGraph::get_in_deg(int v){
	if (v < 0 || v >= max_n_v) return -1;
	int d = 0;
	CUDA_CHECK(cudaMemcpy(&d, h_in.deg + v, sizeof(int), cudaMemcpyDeviceToHost));
	return d;
}

bool SlabGraph::has_edge(int u, int v){ 
	return get_weight(u, v) != -1; 
}
int  SlabGraph::is_valid(int u){ 
	if (u<0 || u>= max_n_v) return 0;
	if (d_ptr == 0) return 0;

	bool v = false;
	CUDA_CHECK(cudaMemcpy(&v, d_valid + u, sizeof(bool), cudaMemcpyDeviceToHost));
	/* DONE */ return v; 
}
int  SlabGraph::get_weight(int u, int v){
	if (u<0 || u>= max_n_v) return -1;
	if (v<0 || v>= max_n_v) return -1;
	if (d_ptr == 0) return -1;
	int val = -1;
	int * d_res;
	cudaMalloc(&d_res, sizeof(int));
	k_get_weights<<<1, 32>>>(h_out, u, v, d_res);
	CUDA_CHECK(cudaGetLastError());
	cudaMemcpy(&val, d_res, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_res);
	return val; 
	/* DONE: 1-warp kernel calling warp_search on the out-list of u */
}
int  SlabGraph::get_in_neighbors(int u, Neighbor* buf, int n){ 
	if (n<=0) return 0;
	if (u<0 || u>= max_n_v) return -1;
	if (d_ptr == 0) return -1;

	Neighbor* d_buf;
	cudaMalloc(&d_buf, sizeof(Neighbor)*n);
	int * d_count;
	cudaMalloc(&d_count, sizeof(int));
	cudaMemset(d_count, 0, sizeof(int)*n);

	/* TODO: k_neighbors on h_in */ return 0; 
}
int  SlabGraph::get_out_neighbors(int u, Neighbor* buf, int n){ /* TODO: k_neighbors on h_out */ return 0; }
int  SlabGraph::edge_list(Edge* buf, int n){ /* TODO */ return 0; }
void SlabGraph::remove_vertex(int u){ /* TODO */ }
