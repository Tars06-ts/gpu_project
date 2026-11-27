#include <cuda.h>
#include <bit>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "graph/csr.cuh"
#include "graph/checks.cuh"
#include "graph/primitives.cuh"
#include "graph/types.cuh"
#include "graph/util.h"

// allocates and fills value -1
static int* alloc_neg1(size_t n){
	size_t bytes = sizeof(int) * (n ? n : 1);
	int* p;
	CUDA_CHECK(cudaMalloc((void**)&p, bytes));
	CUDA_CHECK(cudaMemset(p, 0xFF, bytes));
	return p;
}

// stores true into res if edge u->v in graph
// stores false otherwise
// assumes G is valid ptr
__global__ void d_helper_edge(int u, int v, int* wt, CSRView* G){
	__shared__ volatile bool found;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid == 0){
		found = false;
		*wt = -1;
	}
	// make all threads write false to res first
	__syncthreads();
	int laneId = threadIdx.x % 32;
	if (tid/32 == 0){
		int start = G->out_offset[u] + laneId;
		int end = G->out_offset[u+1];
		while (start-laneId < end){
			if (start < end && G->out_edges[start] == v){
				found = true;
			}
			int m = __ballot_sync(0xFFFFFFFF, found);
			if (m!=0 && __ffs(m)-1==laneId){
				*wt = G->out_weights[start];
			}
			if (m!=0){
				break;
			}
			start += warpSize;
		}
	}else if (tid/32 == 1){
		int start = G->in_offset[v] + laneId;
		int end = G->in_offset[v+1];
		while (start-laneId < end && !found){
			if (start < end && G->in_edges[start] == u){
				found = true;
			}
			int m = __ballot_sync(0xFFFFFFFF, found);
			if (m!=0 && __ffs(m)-1==laneId){
				*wt = G->out_weights[start];
			}
			if (m!=0){
				break;
			}
			start += warpSize;
		}
	}
}

__global__ void d_helper_neighbors(int u, Neighbor* buf, int n_buf,
								   int* neighbors, int* weights,
								   int n){
	int start = blockDim.x * blockIdx.x + threadIdx.x;
	while ((start < n_buf) && (start < n)){
		buf[start] = {neighbors[start], weights[start]};
		start += warpSize;
	}
}

__global__ void d_helper_edge_list(Edge* buf, int n_buf, CSRView* G){
	// each vertex gets its own warp
	int u = G->min_v + (blockDim.x * blockIdx.x + threadIdx.x) / warpSize;
	if (u > G->max_v){
		return;
	}
	while (u < G->max_v){
		int start = min(n_buf, G->out_offset[u]) + (threadIdx.x % 32);
		int end = min(n_buf, G->out_offset[u+1]);
		while (start < end){
			if (G->out_edges[start] != -1){
				// if out_edges[start] == -1, deleted edge
				buf[start] = {u, G->out_edges[start], G->out_weights[start]};
			}
			start += warpSize;
		}
		u += (gridDim.x * blockDim.x) / warpSize;		
	}
}

__global__ void d_helper_remove_vertex(int del_v, int max_v, int min_v,
        int *in_offset, int *in_edges, int *in_weights, int *in_used,
        int *out_offset, int *out_edges, int *out_weights, int *out_used){
	int lane  = threadIdx.x % warpSize;
	int warps = (gridDim.x * blockDim.x) / warpSize;
	for (int u = min_v + (threadIdx.x + blockDim.x * blockIdx.x) / warpSize;
	     u <= max_v; u += warps){
		for (int i = in_offset[u] + lane; i < in_offset[u] + in_used[u]; i += warpSize){
			if (u == del_v || in_edges[i] == del_v){
				in_edges[i] = -1;
				in_weights[i] = -1;
			}
		}
		for (int i = out_offset[u] + lane; i < out_offset[u] + out_used[u]; i += warpSize){
			if (u == del_v || out_edges[i] == del_v){
				out_edges[i] = -1;
				out_weights[i] = -1;
			}
		}
	}
}

__global__ void d_helper_pack_edges(int max_v, int min_v,
									int *edges, int *weights,
									int *offset, int *used,
									bool* valid){
	int u = min_v + (threadIdx.x + blockDim.x * blockIdx.x) / warpSize;
	while (u <= max_v){
		// a warp per vertex
		if (!valid[u]){
			u += (gridDim.x * blockDim.x) / warpSize;
			continue;
		}
		int lane = threadIdx.x % warpSize;
		int start = offset[u];
		int end = offset[u+1];
		int dst = start;
		while (start < end){
			int idx = start + lane;
			bool live = (idx < end) && (edges[idx] != -1);
			uint32_t bits = __ballot_sync(0xFFFFFFFF, live);
			int v,wt;
			if (live){
				v = edges[idx];
				wt = weights[idx];
			}
			__syncwarp();
			if (live){
				int pos = dst + __popc(bits & (1u<<lane)-1);
				edges[pos] = v;
				weights[pos] = wt;
			}
			dst += __popc(bits);
			start += warpSize;
			__syncwarp();
		}
		for (int i=dst+lane;i<end;i+=warpSize){
			edges[i] = -1;
			weights[i] = -1;
		}
		__syncwarp();
		if (lane==0){
			used[u] = dst-start;
		}
		u += (gridDim.x * blockDim.x) / warpSize;
	}
}

void __global__ d_helper_delete_edges(Edge* elist, int n, int* offset, int* edges, int* weights, int* used, bool out_del, int max_n_v){
/*
 *	a thread per edge per direction (in-edge / out-edge) to be deleted
 *	do not decrement 'n' from n_e as some deletions maybe invalid
 */
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n){
		return;
	}
	while (idx < n){
		int u = elist[idx].u;
		int v = elist[idx].v;

		if (u<0||u>=max_n_v||v<0||v>=max_n_v){
			idx += gridDim.x * blockDim.x;
			continue;
		}

		if (!out_del){
			int tmp = u;
			u = v;
			v = tmp;
		}

		for (int i=offset[u];i<offset[u]+used[u];i++){
			if (edges[i]==v){
				edges[i] = -1;
				weights[i] = -1;
				break;
			}
		}
		idx += gridDim.x * blockDim.x;
	}
}


__global__ void d_helper_needed(Edge* ins, int n_ins, bool *valid_ins_pool,
						int *out_needed, int *in_needed, int max_n_v){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	while (idx < n_ins){
		if (valid_ins_pool[idx]){
			int u = ins[idx].u;
			int v = ins[idx].v;
			atomicAdd(&in_needed[v],1);
			atomicAdd(&out_needed[u],1);
		}
		idx += gridDim.x * blockDim.x;
	}
}

__global__ void d_helper_decide_realloc(int max_n_v, bool* valid,
							int* out_offset, int* out_used, int* out_needed,
							int* in_offset, int* in_used, int* in_needed,
							bool* realloc){
	__shared__ bool res;
	res = false;
	__syncthreads();
	// if any vertex needs more space, set realloc and res to true
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	while (idx < max_n_v){
		if (!valid[idx]){
			idx += gridDim.x * blockDim.x;
			continue;
		}
		int out_have = out_offset[idx+1]-out_offset[idx]-out_used[idx];
		if (out_have < out_needed[idx]){
			res = true;
			*realloc = true;
		}
		int in_have = in_offset[idx+1]-in_offset[idx] - in_used[idx];
		if (in_have < in_needed[idx]){
			res = true;
			*realloc = true;
		}
		if (res){
			break;
		}
		idx += gridDim.x * blockDim.x;
	}
}

__global__ void d_helper_compute_degree(int *out_needed, int *n_out_offset, 
								int *out_used, int *in_needed, int *n_in_offset, 
								int *in_used, bool *is_vertex_valid, int max_n_v, int slack){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	while (tid <= max_n_v){
		if (tid==max_n_v || !is_vertex_valid[tid]){
			// invalid vertices and the last index get 0 space
			n_in_offset[tid] = n_out_offset[tid] = 0;
		}else{
			int out_live = out_used[tid]+out_needed[tid];
			n_out_offset[tid] = out_live + max(slack, out_live/2);

			int in_live = in_used[tid]+in_needed[tid];
			n_in_offset[tid] = in_live + max(slack, in_live/2);
		}

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void d_helper_copy_from_old_to_new(int *valid,
							int *edges, int *weights, int *offset,
							int *n_edges, int *n_weights, int *n_offset,
							int max_n_v, CSRView *G){
	// one warp per vertex
	int u = (threadIdx.x + blockDim.x*blockIdx.x) / warpSize;
	while (u < max_n_v){
		int start = (threadIdx.x % warpSize), end = valid[u];
		while (start < end){
			n_edges[n_offset[u]+start] = edges[offset[u]+start];
			n_weights[n_offset[u]+start] = weights[offset[u]+start];

			start += warpSize;
		}

		u += (gridDim.x * blockDim.x) / warpSize;
	}
}

__global__ void d_helper_drop_invalid_ins(Edge *ins, bool *is_valid_ins,
									int n_ins, int max_n_v, 
									bool *is_valid_vertex, bool *is_new_vertex){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	while (tid < n_ins){
		int u = ins[tid].u;
		int v = ins[tid].v;
		bool valid_u = u>=0 && u<max_n_v;
		bool valid_v = v>=0 && v<max_n_v;
		if (valid_u && valid_v){
			is_valid_ins[tid] = true;
			if (!is_valid_vertex[u]){
				is_new_vertex[u] = true;
			}
			if (!is_valid_vertex[v]){
				is_new_vertex[v] = true;
			}
		}else{
			is_valid_ins[tid] = false;
		}
		tid += blockDim.x * gridDim.x;
	}

}

__global__ void d_helper_update_existing_edges(Edge *ins, int n_ins,
									int *edges, int *weights,
									int *offset, bool in, 
									bool *valid_ins_pool,
									bool *is_update){
	// a warp per insertion
	int idx = (threadIdx.x + blockDim.x*blockIdx.x) / warpSize;
	while (idx < n_ins){
		if (!valid_ins_pool[idx]){
			idx += (gridDim.x*blockDim.x)/warpSize;
			continue;
		}
		int u = ins[idx].u;
		int v = ins[idx].v;
		int wt = ins[idx].weight;
		if (in){
			int tmp = u;
			u = v;
			v = tmp;
		}
		bool found = false;
		int lane = threadIdx.x % warpSize;
		int start = offset[u]+lane, end = offset[u+1];
		while (start-lane < end){
			if (start<end && edges[start] == v){
				found = true;
			}
			int m = __ballot_sync(0xFFFFFFFF, found);
			if (m!=0){
				if (__ffs(m)-1 == lane){
					weights[start] = wt;
					is_update[idx] = true;
				}
				break;
			}
			start += warpSize;
		}
		idx += (gridDim.x*blockDim.x)/warpSize;
	}
}

__global__ void d_helper_reserve_slots(Edge *ins, int n_ins,
							bool *valid_ins_pool, int *used,
							int *slots, bool in){
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	while (tid < n_ins){
		if (!valid_ins_pool[tid]){
			tid += gridDim.x*blockDim.x;
			continue;
		}
		int u = (in) ? ins[tid].v : ins[tid].u;
		slots[tid] = atomicAdd(&used[u], 1);
		tid += gridDim.x*blockDim.x;
	}
}

__global__ void d_helper_insert_new_edges(Edge *ins, int n_ins,
						int *offset, int *edges, int *weights,
						bool *valid_ins_pool, bool in, int *slots){
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	while (tid < n_ins){
		if (!valid_ins_pool[tid]){
			tid += gridDim.x*blockDim.x;
			continue;
		}
		int u = ins[tid].u;
		int v = ins[tid].v;
		int wt = ins[tid].weight;
		if (in){
			int tmp = u;
			u = v;
			v = tmp;
		}
		edges[offset[u]+slots[tid]] = v;
		weights[offset[u]+slots[tid]] = wt;

		tid += gridDim.x*blockDim.x;
	}
}

__global__ void d_helper_invalidate_updates(bool *valid_ins_pool,
										bool *is_update, int n_ins){
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	while (idx < n_ins){
		valid_ins_pool[idx] = valid_ins_pool[idx] && !(is_update[idx]);

		idx += gridDim.x*blockDim.x;
	}
}

__global__ void d_helper_activate_vertices(bool* valid, const bool* is_new, int n){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	while (tid<n){
		if (is_new[tid]) valid[tid] = true;
		tid += gridDim.x*blockDim.x;
	}
}

CSR::CSR(int n_v, int max_n_v){
	this->n_v = n_v;
	this->max_n_v = max_n_v;
	this->max_v = n_v-1;
	this->min_v = 0;
	this->n_e = 0;
	valid.resize(max_n_v);
	std::fill(valid.begin(), valid.begin() + n_v, true);
	std::fill(valid.begin() + n_v, valid.end(), false);
	in_offset.assign(max_n_v+1, 0);
	in_edges.clear();
	in_weights.clear();
	in_used.assign(max_n_v, 0);
	out_offset.assign(max_n_v+1, 0);
	out_edges.clear();
	out_weights.clear();
	out_used.assign(max_n_v, 0);
	d_n_e = d_n_v = 0;
	d_in_edges = d_in_offset = d_in_weights = d_in_used = 0;
	d_out_edges = d_out_offset = d_out_weights = d_out_used = 0;
	d_valid = 0;
	d_ptr = 0;
	gpu_dirty = false;
}

CSR::CSR(int n_v, int max_n_v, Edge* edge_list, int n_e){
	this->n_v = n_v;
	this->max_n_v = max_n_v;
	this->max_v = n_v-1;
	this->min_v = 0;
	this->n_e = n_e;
	// all are valid vertices
	valid.resize(max_n_v);
	std::fill(valid.begin(), valid.begin() + n_v, true);
	std::fill(valid.begin() + n_v, valid.end(), false);
	in_offset.assign(max_n_v+1, 0);
	in_edges.resize(n_e);
	in_weights.resize(n_e);
	in_used.assign(max_n_v, 0);
	out_offset.assign(max_n_v+1, 0);
	out_edges.resize(n_e);
	out_weights.resize(n_e);
	out_used.resize(max_n_v);
	for (int i=0;i<n_e;i++){
		int u,v;
		u = edge_list[i].u;
		v = edge_list[i].v;
		in_offset[v+1]++;
		out_offset[u+1]++;
	}
	for (int i=1;i<=max_n_v;i++){
		in_used[i-1] = in_offset[i];
		out_used[i-1] = out_offset[i];
		in_offset[i] += in_offset[i-1];
		out_offset[i] += out_offset[i-1];
	}
	std::vector<int> in_idx(n_v, 0), out_idx(n_v, 0);
	for (int i=0;i<n_e;i++){
		int u,v,wt;
		u = edge_list[i].u;
		v = edge_list[i].v;
		wt = edge_list[i].weight;
		// fill the edges
		out_edges[out_idx[u]+out_offset[u]] = v;
		out_weights[out_idx[u]+out_offset[u]] = wt;
		out_idx[u]++;
		in_edges[in_idx[v]+in_offset[v]] = u;
		in_weights[in_idx[v]+in_offset[v]] = wt;
		in_idx[v]++;
	}
	d_n_e = d_n_v = 0;
	d_in_edges = d_in_offset = d_in_weights = 0;
	d_out_edges = d_out_offset = d_out_weights = 0;
	d_ptr = 0;
	gpu_dirty = false;
}

CSR::~CSR(){ 
	free_gpu(); 
}

int CSR::num_vertices(){
	if (gpu_dirty){
		sync_graph();
	}
	return n_v;
}

int CSR::num_edges(){
	if (gpu_dirty){
		sync_graph();
	}
	return n_e;
}

int CSR::get_in_deg(int v){
	if (v >= max_n_v || v < 0){
		return -1;
	}
	if (gpu_dirty){
		sync_graph();
	}
	return in_used[v];
}

int CSR::get_out_deg(int v){
	if (v >= max_n_v || v < 0){
		return -1;
	}
	if (gpu_dirty){
		sync_graph();
	}
	return out_used[v];
}


bool CSR::has_edge(int u, int v){
	return get_weight(u, v) != -1;
}

int CSR::is_valid(int u){
	if (u > max_v || u < 0){
		return false;
	}
	if (gpu_dirty){
		sync_graph();
	}
	return valid[u];
}

int CSR::get_weight(int u, int v){
	if (d_ptr == 0){
		// exit if invalid d_ptr
		FATAL("Invalid copy of Graph in device. to_gpu() hasn't been called yet, or it failed\n");
		return false;
	}
	// launches a kernel to search for edge
	int *d_wt;
	int h_wt = -1;

	cudaEvent_t e;
	cudaStream_t s;
	cudaEventCreate(&e);
	cudaStreamCreate(&s);

	CUDA_CHECK(cudaMalloc((void**)&d_wt, sizeof(int)));

	d_helper_edge<<<1, 32*2, 0, s>>>(u, v, d_wt, d_ptr);
	CUDA_CHECK(cudaGetLastError());

	cudaEventRecord(e, s);
	cudaEventSynchronize(e);

	CUDA_CHECK(cudaMemcpy(&h_wt, d_wt, sizeof(int), cudaMemcpyDeviceToHost));

	cudaEventDestroy(e);
	cudaStreamDestroy(s);
	cudaFree(d_wt);

	return h_wt;
}


void get_neighbors(int u, Neighbor* buf, int n,
				  int* edges, int* weights, int deg){
	// launches a kernel to search for edge
	cudaStream_t s;
	cudaStreamCreate(&s);

	Neighbor* d_buf;

	CUDA_CHECK(cudaMalloc((void**)&d_buf, sizeof(Neighbor)*n));

	d_helper_neighbors<<<1, 32*2, 0, s>>>(u, d_buf, n,
										  edges, weights, deg);
	CUDA_CHECK(cudaGetLastError());

	CUDA_CHECK(cudaMemcpyAsync(buf, d_buf, sizeof(Neighbor)*n, cudaMemcpyDeviceToHost, s));
	CUDA_CHECK(cudaStreamSynchronize(s));

	cudaStreamDestroy(s);
	cudaFree(&d_buf);
}

int CSR::get_in_neighbors(int u, Neighbor* buf, int n){
	if (n <= 0){
		return 0;
	}
	if (u < 0 || u > max_v){
		FATAL("Invalid vertex %d used in get_in_neighbors\n", u);
		return -1;
	}
	if (d_ptr == 0){
		// exit if invalid d_ptr
		FATAL("Invalid copy of Graph in device. to_gpu() hasn't been called yet, or it failed\n");
		return false;
	}

	get_neighbors(u, buf, n,
				  d_in_edges+in_offset[u],
				  d_in_weights+in_offset[u],
				  in_offset[u+1]-in_offset[u]);

	return (in_offset[u+1]-in_offset[u] > n) ? in_offset[u+1]-in_offset[u] : n;
}


int CSR::get_out_neighbors(int u, Neighbor* buf, int n){
	if (n <= 0){
		return 0;
	}
	if (u < 0 || u >= n_v){
		FATAL("Invalid vertex %d used in get_out_neighbors\n", u);
		return -1;
	}
	if (d_ptr == 0){
		// exit if invalid d_ptr
		FATAL("Invalid copy of Graph in device. to_gpu() hasn't been called yet, or it failed\n");
		return false;
	}

	get_neighbors(u, buf, n,
				  d_out_edges+out_offset[u],
				  d_out_weights+out_offset[u],
				  out_offset[u+1]-out_offset[u]);

	return (out_offset[u+1]-out_offset[u] > n) ? out_offset[u+1]-out_offset[u] : n;
}

int CSR::edge_list(Edge* buf, int n){
	if (n <= 0){
		return 0;
	}
	if (d_ptr == 0){
		// exit if invalid d_ptr
		FATAL("Invalid copy of Graph in device. to_gpu() hasn't been called yet, or it failed\n");
		return false;
	}

	cudaStream_t s;
	cudaStreamCreate(&s);
	
	Edge* d_buf;
	CUDA_CHECK(cudaMalloc((void**)&d_buf, sizeof(Edge)*n));
	
	int tpb = 256;
	int total_threads = n_v * warpSize;
	int blocks = (total_threads + tpb - 1)/ tpb;
	blocks = std::min(blocks, 65535);

	d_helper_edge_list<<<blocks, tpb, 0, s>>>(d_buf, n, d_ptr);

	CUDA_CHECK(cudaGetLastError());

	CUDA_CHECK(cudaMemcpyAsync(buf, d_buf, sizeof(Edge)*std::min(n_e,n), cudaMemcpyDeviceToHost, s));

	CUDA_CHECK(cudaStreamSynchronize(s));

	cudaFree(d_buf);
	cudaStreamDestroy(s);

	return std::min(n_e, n);
}


void CSR::pack_all(cudaStream_t s1, cudaStream_t s2){
	int span = max_v - min_v + 1;
	if (span <= 0) return;
	int blocks, tpb;
	launch_dims((long long)span * warpSize, blocks, tpb);
	d_helper_pack_edges<<<blocks, tpb, 0, s1>>>(max_v, min_v, d_out_edges,
			d_out_weights, d_out_offset, d_out_used, d_valid);
	CUDA_CHECK(cudaGetLastError());
	d_helper_pack_edges<<<blocks, tpb, 0, s2>>>(max_v, min_v, d_in_edges,
			d_in_weights, d_in_offset, d_in_used, d_valid);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaStreamSynchronize(s1));
	CUDA_CHECK(cudaStreamSynchronize(s2));
}

void CSR::remove_vertex(int u){
	if (d_ptr == 0){
		FATAL("Invalid copy of Graph in device. to_gpu() hasn't been called yet, or it failed\n");
		return;
	}
	if (u < 0 || u >= max_n_v) return;

	// check if v is already removed in device instead of syncing the whole graph
	bool is_v = false;
	CUDA_CHECK(cudaMemcpy(&is_v, d_valid + u, sizeof(bool), cudaMemcpyDeviceToHost));
	if (!is_v) return;

	pull_scalars();
	cudaStream_t s1, s2;
	CUDA_CHECK(cudaStreamCreate(&s1));
	CUDA_CHECK(cudaStreamCreate(&s2));

	// convert every edge touching u into tombstone
	int blocks, tpb;
	launch_dims((long long)(max_v - min_v + 1) * warpSize, blocks, tpb);
	d_helper_remove_vertex<<<blocks, tpb, 0, s1>>>(u, max_v, min_v,
			d_in_offset, d_in_edges, d_in_weights, d_in_used,
			d_out_offset, d_out_edges, d_out_weights, d_out_used);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaStreamSynchronize(s1));

	// pack while u is still marked valid, so its used counts become 0
	pack_all(s1, s2);

	// now mark u to be invalid
	CUDA_CHECK(cudaMemset(d_valid + u, 0, sizeof(bool)));

	// recompute the scalars from the device state
	BoolStats st = bool_stats(d_valid, max_n_v, s1);
	n_v = st.count;
	// min_v = max_v = -1 if the graph is now empty
	min_v = st.min_idx;       	
	max_v = st.max_idx;
	n_e = sum_int(d_out_used, max_n_v, s1);
	push_scalars();
	gpu_dirty = true;

	CUDA_CHECK(cudaStreamDestroy(s1));
	CUDA_CHECK(cudaStreamDestroy(s2));
}

void CSR::apply(Edge* ins, int n_ins, 
				Edge* del, int n_del){
	if (d_ptr == 0){
		// exit if invalid d_ptr
		FATAL("Invalid copy of Graph in device. to_gpu() hasn't been called yet, or it failed\n");
		return;
	}
	// update the host if any changes in device
	pull_scalars(); 

	cudaStream_t s1,s2;
	cudaStreamCreate(&s1);
	cudaStreamCreate(&s2);

	if (n_del > 0) apply_deletions(del, n_del, s1, s2);
	if (n_ins > 0) apply_insertions(ins, n_ins, s1, s2);

	push_scalars();
	// gpu copy has updated graph
	gpu_dirty = true;

	CUDA_CHECK(cudaStreamDestroy(s1));
	CUDA_CHECK(cudaStreamDestroy(s2));
}

void CSR::apply_deletions(Edge* del, int n_del, cudaStream_t s1, cudaStream_t s2){
	int blocks, tpb;
	launch_dims(n_del, blocks, tpb);
	d_helper_delete_edges<<<blocks, tpb, 0, s1>>>(del, n_del, d_in_offset,
			d_in_edges, d_in_weights, d_in_used, false, max_n_v);
	CUDA_CHECK(cudaGetLastError());
	d_helper_delete_edges<<<blocks, tpb, 0, s2>>>(del, n_del, d_out_offset,
			d_out_edges, d_out_weights, d_out_used, true, max_n_v);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaStreamSynchronize(s1));
	CUDA_CHECK(cudaStreamSynchronize(s2));

	// pack live edges to the front of each segment
	pack_all(s1, s2);
	// deletions may have been invalid, so recount from the packed degrees
	n_e = sum_int(d_out_used, max_n_v, s1);
}

void CSR::apply_insertions(Edge* ins, int n_ins, cudaStream_t s1, cudaStream_t s2){
	InsertCtx c = {};
	c.ins = ins;
	c.n_ins = n_ins;

	ins_filter(c, s1);                          // drops invalid, deduplicate, activate vertices
	if (ins_count_demand(c, s1)){               // checks if any vertex require more room
		ins_rebuild(c, s1, s2);                 // reallocates new edges, weights, offsets and swaps pointers
	}
	ins_update_existing(c, s1, s2);             // existing edges: weight update only
	ins_write_new(c, s1, s2);                   // reserve slots, write, count for new edges

	cudaFree(c.pool);       cudaFree(c.new_vertex); cudaFree(c.is_update);
	cudaFree(c.out_needed); cudaFree(c.in_needed);
	cudaFree(c.out_slots);  cudaFree(c.in_slots);
}

void CSR::ins_filter(InsertCtx& c, cudaStream_t s){
	CUDA_CHECK(cudaMalloc(&c.pool, sizeof(bool) * c.n_ins));
	CUDA_CHECK(cudaMemset(c.pool, 0, sizeof(bool) * c.n_ins));
	CUDA_CHECK(cudaMalloc(&c.new_vertex, sizeof(bool) * max_n_v));
	CUDA_CHECK(cudaMemset(c.new_vertex, 0, sizeof(bool) * max_n_v));

	int blocks, tpb;
	launch_dims(c.n_ins, blocks, tpb);
	d_helper_drop_invalid_ins<<<blocks, tpb, 0, s>>>(c.ins, c.pool, c.n_ins,
			max_n_v, d_valid, c.new_vertex);
	CUDA_CHECK(cudaGetLastError());

	// keep only the latest insertion of each (u,v)
	dedup_insertions(c.ins, c.pool, c.n_ins, s);

	// newly referenced vertices: count them
	// update min_v, max_v
	BoolStats st = bool_stats(c.new_vertex, max_n_v, s);
	if (st.count > 0){
		n_v += st.count;
		max_v = max(max_v, st.max_idx);
		min_v = min(min_v, st.min_idx);
		launch_dims(max_n_v, blocks, tpb);
		d_helper_activate_vertices<<<blocks, tpb, 0, s>>>(d_valid, c.new_vertex, max_n_v);
		CUDA_CHECK(cudaGetLastError());
	}
}

bool CSR::ins_count_demand(InsertCtx& c, cudaStream_t s){
	size_t bytes = sizeof(int) * max_n_v;
	CUDA_CHECK(cudaMalloc(&c.out_needed, bytes));
	CUDA_CHECK(cudaMemset(c.out_needed, 0, bytes));
	CUDA_CHECK(cudaMalloc(&c.in_needed, bytes));
	CUDA_CHECK(cudaMemset(c.in_needed, 0, bytes));

	int blocks, tpb;
	launch_dims(c.n_ins, blocks, tpb);
	d_helper_needed<<<blocks, tpb, 0, s>>>(c.ins, c.n_ins, c.pool,
			c.out_needed, c.in_needed, max_n_v);
	CUDA_CHECK(cudaGetLastError());

	bool* d_flag;
	CUDA_CHECK(cudaMalloc(&d_flag, sizeof(bool)));
	CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(bool)));
	launch_dims(max_n_v, blocks, tpb);
	d_helper_decide_realloc<<<blocks, tpb, 0, s>>>(max_n_v, d_valid,
			d_out_offset, d_out_used, c.out_needed,
			d_in_offset,  d_in_used,  c.in_needed, d_flag);
	CUDA_CHECK(cudaGetLastError());

	bool need = false;
	CUDA_CHECK(cudaMemcpyAsync(&need, d_flag, sizeof(bool), cudaMemcpyDeviceToHost, s));
	CUDA_CHECK(cudaStreamSynchronize(s));
	cudaFree(d_flag);
	return need;
}

void CSR::ins_rebuild(InsertCtx& c, cudaStream_t s1, cudaStream_t s2){
	const int V = max_n_v;
	int *n_out_off, *n_in_off;
	CUDA_CHECK(cudaMalloc(&n_out_off, sizeof(int) * (V + 1)));
	CUDA_CHECK(cudaMalloc(&n_in_off,  sizeof(int) * (V + 1)));

	// finds per-vertex capacity, then prefix sum into offsets
	int blocks, tpb;
	launch_dims(V + 1, blocks, tpb);
	d_helper_compute_degree<<<blocks, tpb, 0, s1>>>(c.out_needed, n_out_off,
			d_out_used, c.in_needed, n_in_off, d_in_used, d_valid, V, MIN_SLACK);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaStreamSynchronize(s1));
	exclusive_scan_inplace(n_out_off, V + 1, s1);
	exclusive_scan_inplace(n_in_off,  V + 1, s2);

	int e_out = 0, e_in = 0;
	CUDA_CHECK(cudaMemcpy(&e_out, n_out_off + V, sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&e_in,  n_in_off  + V, sizeof(int), cudaMemcpyDeviceToHost));

	// new arrays, pre-filled with -1
	int *n_out_e = alloc_neg1(e_out), *n_out_w = alloc_neg1(e_out);
	int *n_in_e  = alloc_neg1(e_in),  *n_in_w  = alloc_neg1(e_in);

	// copy live edges: one warp per vertex
	launch_dims((long long)V * warpSize, blocks, tpb);
	d_helper_copy_from_old_to_new<<<blocks, tpb, 0, s1>>>(d_out_used, d_out_edges,
			d_out_weights, d_out_offset, n_out_e, n_out_w, n_out_off, V, d_ptr);
	CUDA_CHECK(cudaGetLastError());
	d_helper_copy_from_old_to_new<<<blocks, tpb, 0, s2>>>(d_in_used, d_in_edges,
			d_in_weights, d_in_offset, n_in_e, n_in_w, n_in_off, V, d_ptr);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaStreamSynchronize(s1));
	CUDA_CHECK(cudaStreamSynchronize(s2));

	// point the device struct at the NEW arrays
	push_ptr(offsetof(CSRView, in_offset),   n_in_off);
	push_ptr(offsetof(CSRView, in_edges),    n_in_e);
	push_ptr(offsetof(CSRView, in_weights),  n_in_w);
	push_ptr(offsetof(CSRView, out_offset),  n_out_off);
	push_ptr(offsetof(CSRView, out_edges),   n_out_e);
	push_ptr(offsetof(CSRView, out_weights), n_out_w);

	// free the old arrays and adopt the new ones on the host side
	cudaFree(d_in_offset);  cudaFree(d_in_edges);  cudaFree(d_in_weights);
	cudaFree(d_out_offset); cudaFree(d_out_edges); cudaFree(d_out_weights);
	d_in_offset = n_in_off;   d_in_edges = n_in_e;   d_in_weights = n_in_w;
	d_out_offset = n_out_off; d_out_edges = n_out_e; d_out_weights = n_out_w;
}

void CSR::ins_update_existing(InsertCtx& c, cudaStream_t s1, cudaStream_t s2){
	CUDA_CHECK(cudaMalloc(&c.is_update, sizeof(bool) * c.n_ins));
	CUDA_CHECK(cudaMemset(c.is_update, 0, sizeof(bool) * c.n_ins));

	int blocks, tpb;
	launch_dims((long long)c.n_ins * warpSize, blocks, tpb);     // a warp per insertion
	d_helper_update_existing_edges<<<blocks, tpb, 0, s1>>>(c.ins, c.n_ins,
			d_in_edges, d_in_weights, d_in_offset, true,
			c.pool, c.is_update);
	CUDA_CHECK(cudaGetLastError());
	d_helper_update_existing_edges<<<blocks, tpb, 0, s2>>>(c.ins, c.n_ins,
			d_out_edges, d_out_weights, d_out_offset, false,
			c.pool, c.is_update);
	CUDA_CHECK(cudaGetLastError());

	CUDA_CHECK(cudaStreamSynchronize(s1));
	CUDA_CHECK(cudaStreamSynchronize(s2));

	// updates are done; what remains in the pool are genuinely new edges
	launch_dims(c.n_ins, blocks, tpb);
	d_helper_invalidate_updates<<<blocks, tpb, 0, s1>>>(c.pool, c.is_update, c.n_ins);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaStreamSynchronize(s1));
}

void CSR::ins_write_new(InsertCtx& c, cudaStream_t s1, cudaStream_t s2){
	CUDA_CHECK(cudaMalloc(&c.out_slots, sizeof(int) * c.n_ins));
	CUDA_CHECK(cudaMalloc(&c.in_slots,  sizeof(int) * c.n_ins));

	int blocks, tpb;
	launch_dims(c.n_ins, blocks, tpb);

	// claim a unique slot in each vertex's segment (this also advances *_used)
	d_helper_reserve_slots<<<blocks, tpb, 0, s1>>>(c.ins, c.n_ins, c.pool,
			d_out_used, c.out_slots, false);
	CUDA_CHECK(cudaGetLastError());
	d_helper_reserve_slots<<<blocks, tpb, 0, s2>>>(c.ins, c.n_ins, c.pool,
			d_in_used, c.in_slots, true);
	CUDA_CHECK(cudaGetLastError());

	CUDA_CHECK(cudaStreamSynchronize(s1));
	CUDA_CHECK(cudaStreamSynchronize(s2));

	// write the edges into their slots
	d_helper_insert_new_edges<<<blocks, tpb, 0, s1>>>(c.ins, c.n_ins, d_out_offset,
			d_out_edges, d_out_weights, c.pool, false, c.out_slots);
	CUDA_CHECK(cudaGetLastError());
	d_helper_insert_new_edges<<<blocks, tpb, 0, s2>>>(c.ins, c.n_ins, d_in_offset,
			d_in_edges, d_in_weights, c.pool, true, c.in_slots);
	CUDA_CHECK(cudaGetLastError());

	CUDA_CHECK(cudaStreamSynchronize(s1));
	CUDA_CHECK(cudaStreamSynchronize(s2));

	// what is left in the pool is exactly the set of new edges
	n_e += count_true(c.pool, c.n_ins, s1);
}

CSRView* CSR::to_gpu(){
	// copies object to gpu as CSRView and
	// returns device pointer
	// returns 0 if something fails
	if (d_ptr != 0){
		// already uploaded
		return d_ptr;        	
	}
	const size_t V = max_n_v;

	// allocate (at least 1 element so no pointer is null) and upload into device
	auto up = [](int** dst, const std::vector<int>& src, size_t n){
		if (cudaMalloc((void**)dst, sizeof(int)*(n ? n : 1)) != cudaSuccess){
			*dst = 0;
			return false;
		}
		if (n == 0) return true;
		return cudaMemcpy(*dst, src.data(), sizeof(int)*n,
		                  cudaMemcpyHostToDevice) == cudaSuccess;
	};

	bool ok = up(&d_in_offset,  in_offset,  V+1)
	       && up(&d_in_used,    in_used,    V)
	       && up(&d_in_edges,   in_edges,   in_edges.size())
	       && up(&d_in_weights, in_weights, in_weights.size())
	       && up(&d_out_offset, out_offset, V+1)
	       && up(&d_out_used,   out_used,   V)
	       && up(&d_out_edges,  out_edges,  out_edges.size())
	       && up(&d_out_weights,out_weights,out_weights.size());

	if (ok){
		// vector<bool> is bit-packed, so expand it to one byte per vertex
		std::vector<char> hv(valid.begin(), valid.end());
		ok = cudaMalloc((void**)&d_valid, V) == cudaSuccess
		  && cudaMemcpy(d_valid, hv.data(), V, cudaMemcpyHostToDevice) == cudaSuccess;
	}
	if (ok){
		ok = cudaMalloc((void**)&d_ptr, sizeof(CSRView)) == cudaSuccess;
	}
	if (!ok){
		free_gpu();
		return 0;
	}

	CSRView h;
	h.max_n_v = max_n_v;  h.n_v = n_v;  h.n_e = n_e;
	h.max_v = max_v;      h.min_v = min_v;
	h.valid = d_valid;
	h.in_offset = d_in_offset;   h.in_edges = d_in_edges;
	h.in_weights = d_in_weights; h.in_used = d_in_used;
	h.out_offset = d_out_offset;   h.out_edges = d_out_edges;
	h.out_weights = d_out_weights; h.out_used = d_out_used;
	if (cudaMemcpy(d_ptr, &h, sizeof(h), cudaMemcpyHostToDevice) != cudaSuccess){
		free_gpu();
		return 0;
	}

	d_n_v = (int*)((char*)d_ptr + offsetof(CSRView, n_v));
	d_n_e = (int*)((char*)d_ptr + offsetof(CSRView, n_e));
	gpu_dirty = false;
	return d_ptr;
	return 0;
}

CSRView* CSR::view(){
	return d_ptr;
}

void CSR::free_gpu(){
	// frees d_ptr
	if (d_ptr != 0 && gpu_dirty){
		sync_graph();                 // keep the host copy so nothing is lost
	}
	cudaFree(d_in_offset);		cudaFree(d_in_edges);
	cudaFree(d_in_weights);		cudaFree(d_in_used);
	cudaFree(d_out_offset);		cudaFree(d_out_edges);
	cudaFree(d_out_weights);	cudaFree(d_out_used);
	cudaFree(d_valid);
	cudaFree(d_ptr);           	
	d_in_offset = d_in_edges = d_in_weights = d_in_used = 0;
	d_out_offset = d_out_edges = d_out_weights = d_out_used = 0;
	d_valid = 0;
	d_ptr = 0;
	d_n_v = d_n_e = 0;
	gpu_dirty = false;
}

static_assert(offsetof(CSRView, min_v) == offsetof(CSRView, n_v) + 3*sizeof(int),
              "push/pull_scalars expect n_v, n_e, max_v, min_v to be contiguous");

void CSR::push_scalars(){
	int h[4] = {n_v, n_e, max_v, min_v};
	CUDA_CHECK(cudaMemcpy((char*)d_ptr + offsetof(CSRView, n_v), h, sizeof(h),
	                      cudaMemcpyHostToDevice));
}

void CSR::pull_scalars(){
	int h[4];
	CUDA_CHECK(cudaMemcpy(h, (char*)d_ptr + offsetof(CSRView, n_v), sizeof(h),
	                      cudaMemcpyDeviceToHost));
	n_v = h[0]; n_e = h[1]; max_v = h[2]; min_v = h[3];
}

// overwrite one pointer field of the device struct (CSRView)
void CSR::push_ptr(size_t field_offset, int* p){
	CUDA_CHECK(cudaMemcpy((char*)d_ptr + field_offset, &p, sizeof(int*),
	                      cudaMemcpyHostToDevice));
}
void CSR::sync_graph(){
	// copies into host from device
	if (d_ptr == 0) return;
	CUDA_CHECK(cudaDeviceSynchronize());
	pull_scalars();

	std::vector<char> hv(max_n_v);
	CUDA_CHECK(cudaMemcpy(hv.data(), d_valid, max_n_v, cudaMemcpyDeviceToHost));
	valid.assign(hv.begin(), hv.end());

	auto down = [](std::vector<int>& dst, const int* src, size_t n){
		dst.resize(n);
		if (n) CUDA_CHECK(cudaMemcpy(dst.data(), src, sizeof(int)*n,
		                             cudaMemcpyDeviceToHost));
	};
	down(in_offset,  d_in_offset,  max_n_v + 1);
	down(out_offset, d_out_offset, max_n_v + 1);
	down(in_used,    d_in_used,    max_n_v);
	down(out_used,   d_out_used,   max_n_v);
	down(in_edges,    d_in_edges,    in_offset[max_n_v]);
	down(in_weights,  d_in_weights,  in_offset[max_n_v]);
	down(out_edges,   d_out_edges,   out_offset[max_n_v]);
	down(out_weights, d_out_weights, out_offset[max_n_v]);
	gpu_dirty = false;
}



