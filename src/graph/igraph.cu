#include <cuda.h>
#include "graph/igraph.cuh"
#include "graph/checks.cuh"
#include "graph/exclusive_scan.cuh"
#include <bit>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

int next_power_of_2(int u){
    if (u <= 1) return 1;
    // Prevent signed overflow for values > 2^30 (1,073,741,824)
    if (u > 1073741824) return 1073741824; 

    unsigned int v = static_cast<unsigned int>(u - 1);
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return static_cast<int>(v + 1);	
}

template <typename... Args>
int max(int first, Args... rest) {
    int res = first;
    ((res = (rest > res) ? rest : res), ...);
    return res;
}

template <typename... Args>
int min(int first, Args... rest) {
    int res = first;
    ((res = (rest < res) ? rest : res), ...);
    return res;
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
	d_in_edges = d_in_offset = d_in_weights = 0;
	d_out_edges = d_out_offset = d_out_weights = 0;
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
	if (v >= n_v || v < 0){
		return -1;
	}
	if (gpu_dirty){
		sync_graph();
	}
	return in_used[v];
}

int CSR::get_out_deg(int v){
	if (v >= n_v || v < 0){
		return -1;
	}
	if (gpu_dirty){
		sync_graph();
	}
	return out_used[v];
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
		while (start < end){
			if (G->out_edges[start] == v){
				found = true;
				*wt = G->out_weights[start];
			}
			if (__any_sync(0xFFFFFFFF, found)){
				break;
			}
			start += warpSize;
		}
	}else if (tid/32 == 1){
		int start = G->in_offset[v] + laneId;
		int end = G->in_offset[v+1];
		while (start < end && !found){
			if (G->in_edges[start] == u){
				found = true;
				*wt = G->in_weights[start];
			}
			if (__any_sync(0xFFFFFFFF, found)){
				break;
			}
			start += warpSize;
		}
	}
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


__global__ void d_helper_neighbors(int u, Neighbor* buf, int n_buf,
								   int* neighbors, int* weights,
								   int n){
	int start = blockDim.x * blockIdx.x + threadIdx.x;
	while ((start < n_buf) && (start < n)){
		buf[start] = {neighbors[start], weights[start]};
		start += warpSize;
	}
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

__global__ void d_helper_remove_vertex(int del_v, CSRView* G, int* n_max_v, int* n_min_v){
	// assign one warp per vertex
	// if target_vertex == del_v
	//		target vertices = -1 for in_edges where
	// if u == del_v
	//		set all to -1 for out_edges
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid == 0){
		G->n_v--;
		G->valid[del_v] = false;
		G->out_used[del_v] = 0;
	}
	int u = G->min_v + (tid / warpSize);
	if (u > G->max_v){
		return;
	}
	while (u <= G->max_v){
		int start = G->out_offset[u] + (threadIdx.x % 32);
		int end = G->out_offset[u+1];
		bool found = false;
		while (start < end){
			if (u == del_v){
				// delete all targets
				G->out_edges[start] = -1;
			}else{
				// delete if equals v and return as
				// simple graph
				if (G->in_edges[start] == del_v){
					G->in_edges[start] = -1;
					G->in_used -= 1;
					found = true;
				}
				// whole warp breaks if found
				if (__any_sync(0xFFFFFFFF, found)){
					break;
				}
			}
			start += warpSize;
		}
		u += (gridDim.x * blockDim.x) / warpSize;		
	}
	if (tid / warpSize == 0){
		// change min_v and max_v if required
		bool found = false;
		if (del_v == G->min_v){
			int start = G->min_v + 1;
			while (start <= G->max_v){
				if (G->valid[start]){
					G->min_v = start;
					*n_min_v = start;
				}
				if (__any_sync(0xFFFFFFFF, found){
					break;
				}
				start += warpSize;
			}
			if (!found){
				G->min_v = -1;
				*n_min_v = -1;
			}
		}
		if (del_v == G->max_v){
			int start = G->max_v - 1;
			while (start >= G->min_v){
				if (G->valid[start]){
					G->max_v = start;
					*n_max_v = start;
				}
				if (__any_sync(0xFFFFFFFF, found){
					break;
				}
				start -= warpSize;
			}
			if (!found){
				G->max_v = -1;
				*n_max_v = -1;
			}
		}
	}
}

void CSR::remove_vertex(int u){
	if (!is_valid(u)){
		return;
	}
	// launch a kernel to invalidate edges for in_edges and out_edges
	// make in_used -= (number of vertices having in-edge from u)
	if (u <= 0){
		return;
	}
	if (d_ptr == 0){
		// exit if invalid d_ptr
		FATAL("Invalid copy of Graph in device. to_gpu() hasn't been called yet, or it failed\n");
		return;
	}

	cudaStream_t s;
	cudaStreamCreate(&s);
	
	int tpb = 256;
	int total_threads = n_v * warpSize;
	int blocks = (total_threads + tpb - 1)/ tpb;
	blocks = min(blocks, 65535);


	int *d_max_v, *d_min_v;

	CUDA_CHECK(cudaMalloc(&d_max_v, sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_max_v, sizeof(int)));

	d_helper_remove_vertex<<<blocks, tpb, 0, s>>>(u, d_ptr, d_max_v, d_min_v);
	CUDA_CHECK(cudaGetLastError());

	CUDA_CHECK(cudaMemcpyAsync(&max_v, d_max_v, sizeof(int), cudaMemcpyDeviceToHost, s));
	CUDA_CHECK(cudaMemcpyAsync(&min_v, d_min_v, sizeof(int), cudaMemcpyDeviceToHost, s));

	cudaStreamSynchronize(s);

	cudaStreamDestroy(s);
}

void __global__ d_helper_delete_edges(Edge* elist, int n, int* offset, int* edges, int* weights, int* used, bool out_del){
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

		if (!out_del){
			int tmp = u;
			u = v;
			v = tmp;
		}

		for (int i=offset[u];i<offset[u+1];i++){
			if (edges[i]==v){
				edges[i] = -1;
				weights[i] = -1;
				break;
			}
		}
		idx += gridDim.x * blockDim.x;
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
			continue;
		}
		int start = offset[u];
		int end = offset[u+1];
		int pack_idx = start;
		int base = 0;
		while (start < end){
			int lane = threadIdx.x % warpSize;
			int idx = start + lane;
			bool live = (idx < end) && (edges[idx] != -1);
			uint32_t bits = __ballot_sync(0xFFFFFFFF, live);
			if (live){
				int v = edges[idx];
				int wt = weights[idx];
			}
			__syncwarp();
			if (live){
				edges[pack_idx + __popc(bits & (1u<<lane)-1)] = v;
				weights[pack_idx + __popc(bits & (1u<<lane)-1)] = wt;
				edges[idx] = -1;
				weights[idx] = -1;
			}
			base += __popc(bits);
			pack_idx += __popc(bits);
			start += warpSize;
		}
		used[u] = base;
		u += (gridDim.x * blockDim.x) / warpSize;
	}
}

__global__ void d_helper_needed(Edge* ins, int n_ins, 
						int *out_needed, int *in_needed, int max_n_v){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	while (idx < n_ins){
		int u = ins[idx].u;
		int v = ins[idx].v;
		if (u < max_n_v && v < max_n_v){
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
	*realloc = false;
	__syncthreads();
	// if any vertex needs more space, set realloc and res to true
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	while (idx < max_n_v){
		if (!valid[idx]){
			continue;
		}
		int out_have = out_offset[idx+1]-out_offset[idx] - out_used[idx];
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

__global__ void d_helper_compute_offset(int *out_needed, int *n_out_offset, 
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

__global__ void d_helper_copy_from_old_to_new(int e_len,
							int *valid,
							int *edges, int *weights, int *offset,
							int *n_edges, int *n_weights, int *n_offset,
							int max_n_v, CSRView *G, bool in){
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
	if (in){
		G->in_offset = offset;
		G->in_edges = edges;
		G->in_weights = weights;
	}else{
		G->out_offset = offset;
		G->out_edges = edges;
		G->out_weights = weights;
	}
}

__global__ void d_helper_drop_invalid_ins(Edge *ins, bool *is_valid,
									int n_ins, int max_n_v){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	while (tid < n_ins){
		int u = ins[tid].u;
		int v = ins[tid].v;
		if (u>=0 && u<max_n_v && v>=0 && v<max_n_v){
			is_valid[tid] = true;
		}else{
			is_valid[tid] = false;
		}
		tid += blockDim.x * gridDim.x;
	}

}

__global__ void d_helper_update_existing_edges(Edge *ins, int n_ins,
									int *edges, int *weights,
									int *offset, bool in, 
									bool* valid_ins_pool,
									bool* is_valid_vertex){
	// a warp per insertion
	int idx = (threadIdx.x + blockDim.x*blockIdx.x) / warpSize;
	while (idx < n_ins){
		if (!valid_ins_pool[idx]){
			idx += (gridDim.x*blockDim.x)/warpSize;
			continue;
		}
		int u = ins[idx].u;
		int v = ins[idx].v;
		if (!is_valid_vertex[u]){
			continue;
		}
		int wt = ins[idx].weight;
		if (in){
			int tmp = u;
			u = v;
			v = tmp;
		}
		bool found = false;
		int start = offset[u]+(threadIdx.x%warpSize), end = offset[u+1];
		while (start < end){
			if (edges[start] == v){
				weights[start] = wt;
				valid_ins_pool[idx] = false;
				found = true;
			}
			if (__any_sync(0xFFFFFFFF, found)){
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
						bool *valid_ins_pool, bool in, int *slots,
						bool *valid_vertex, int *n_max_v, int *n_min_v,
						CSRView *G){
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	while (tid < n_ins){
		if (!valid_ins_pool[tid]){
			tid += gridDim.x*blockDim.x;
			continue;
		}
		int u = ins[tid].u;
		int v = ins[tid].v;
		int wt = ins[tid].weight;
		int count = __popc(__activemask());
		if (threadIdx.x % warpSize == 0){
			atomicAdd(&G->n_e, count);
		}
		if (!valid_vertex[u]){
			atomicAdd(&G->n_v, 1);
		}
		if (!valid_vertex[v]){
			atomicAdd(&G->n_v, 1);
		}
		valid_vertex[u] = valid_vertex[v] = true;
		if (u > v){
			atomicMax(n_max_v, u);
			atomicMin(n_min_v, v);
		}else{
			atomicMax(n_max_v, v);
			atomicMin(n_min_v, u);
		}
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

void CSR::apply(Edge* ins, int n_ins, 
				Edge* del, int n_del){
	if (d_ptr == 0){
		// exit if invalid d_ptr
		FATAL("Invalid copy of Graph in device. to_gpu() hasn't been called yet, or it failed\n");
		return;
	}
	cudaStream_t s1;
	cudaStreamCreate(&s1);
	cudaStream_t s2;
	cudaStreamCreate(&s2);


	if (n_del != 0){
		// launches a kernel for deletion
		int tpb = min(((n_del+warpSize-1)/warpSize)*warpSize, 1024);
		int blocks = max(1, (n_del+tpb-1)/tpb);
		// for in_edges
		d_helper_delete_edges<<<blocks, tpb, 0, s1>>>(del, n_del, 
						d_in_offset, d_in_edges,
						d_in_weights, d_in_used,
						false);
		CUDA_CHECK(cudaGetLastError());
		// for out_edges
		d_helper_delete_edges<<<blocks, tpb, 0, s2>>>(del, n_del, 
						d_out_offset, d_out_edges,
						d_out_weights, d_out_used,
						true);
		CUDA_CHECK(cudaGetLastError());

		cudaStreamSynchronize(s1);
		cudaStreamSynchronize(s2);

		// *_used, n_e are in consistent
		// pack the used edges together
		int t = (max_v-min_v+warpSize / warpSize) * warpSize;
		tpb = min(t, 1024);
		blocks = max(1, (max_v-min_v+1+warpSize-1)/warpSize);
		d_helper_pack_edges<<<blocks, tpb, 0, s1>>>(max_v, min_v, 
							d_out_edges, d_out_weights, 
							d_out_offset, d_out_used, d_valid);
		CUDA_CHECK(cudaGetLastError());
		d_helper_pack_edges<<<blocks, tpb, 0, s2>>>(max_v, min_v, 
							d_in_edges, d_in_weights, 
							d_in_offset, d_in_used, d_valid);
		CUDA_CHECK(cudaGetLastError());

		cudaStreamSynchronize(s1);
		cudaStreamSynchronize(s2);

	}
	if (n_ins != 0){
		// launches a kernel for insertion
		/*
		 *	count the number of extras required for each vertex
		 *	if space isn't available
		 *		reallocate the entire *_edges, *_weights
		 *		only copy the valid edges
		 *		change the offset as required
		 *	now fill in the insertions
		 *
		 */
		int *d_out_needed, *d_in_needed;
		CUDA_CHECK(cudaMallocAsync(&d_out_needed, sizeof(int)*max_n_v, s1));
		CUDA_CHECK(cudaMallocAsync(&d_in_needed, sizeof(int)*max_n_v, s2));

		int t = ((n_ins + warpSize - 1)/warpSize)*warpSize;
		int tpb = min(t, 1024);
		int blocks = max(1, (t+tpb-1)/tpb);

		cudaStreamSynchronize(s1);
		cudaStreamSynchronize(s2);

		d_helper_needed<<<blocks, tpb, 0, s1>>>(ins, n_ins, 
							d_out_needed, d_in_needed, max_n_v);
		CUDA_CHECK(cudaGetLastError());

		// compute (out_offset[u+1]-out_offset[u]-out_used[u]) = free[u]
		// set realloc = true if free[u] < needed[u] for all u
		bool* d_realloc = 0;
		CUDA_CHECK(cudaMallocAsync(&d_realloc, sizeof(bool), s2));

		cudaStreamSynchronize(s2);

		t = (((max_n_v + warpSize - 1)/warpSize)*warpSize;
		tpb = min(t, 1024);
		blocks = max(1, (t+tpb-1)/tpb);
		d_helper_decide_realloc<<<blocks,tpb,0,s1>>>(max_n_v, d_valid, 
								d_out_offset, d_out_used, d_out_needed,
								d_in_offset, d_in_used, d_in_needed, d_realloc);
		CUDA_CHECK(cudaGetLastError());

		bool realloc = false;
		CUDA_CHECK(cudaMemcpy(&realloc, d_realloc, sizeof(bool), cudaMemcpyDeviceToHost));

		if (realloc){
			/*
			 *
			 *	a kernel to compute the new *_offset using exclusive scan
			 *		and the new length of *_edges, *_weights
			 *	allocate on device n_e memory
			 *	a kernel to fill in the old edges + weights into new copy
			 *	free the old copies in device
			 *	change d_* to appropriate varaibles ?
			 *
			 */
			// give an extra of minimum 20 spaces per valid vertex
			int *d_n_out_offset, *d_n_in_offset;
			CUDA_CHECK(cudaMalloc(&d_n_out_offset, sizeof(int)*(max_n_v+1)));
			CUDA_CHECK(cudaMalloc(&d_n_in_offset, sizeof(int)*(max_n_v+1)));

			int t = (((max_n_v+1) + warpSize - 1)/warpSize)*warpSize;
			int tpb = min(t, 1024);
			int blocks = max(1, (t+tpb-1)/tpb);
			d_helper_compute_offset<<<blocks, tpb, 0, s1>>>(d_out_needed, 
									d_n_out_offset, d_out_used, d_in_needed, 
									d_n_in_offset, d_in_used, d_valid, 
									max_n_v, MIN_SLACK);
			CUDA_CHECK(cudaGetLastError());

			cudaStreamSynchronize(s1);
			exclusive_scan_inplace(d_n_out_offset, max_n_v+1, s1);
			exclusive_scan_inplace(d_n_in_offset, max_n_v+1, s2);
			
			// copy from old to new
			int *d_n_out_edges, *d_n_in_edges, *d_n_out_weights, *d_n_in_weights;
			// get the length of the weights array
			int e_len = 0;
			CUDA_CHECK(cudaMemcpy(&e_len, d_n_in_offset+max_n_v, sizeof(int), cudaMemcpyDeviceToHost));
			
			cudaStreamSynchronize(s1);
			cudaStreamSynchronize(s2);

			cudaStream_t s3,s4;
			cudaStreamCreate(&s3);
			cudaStreamCreate(&s4);

			// allocate space for the new arrays
			CUDA_CHECK(cudaMallocAsync(&d_n_out_edges, sizeof(int)*e_len, s1));
			CUDA_CHECK(cudaMallocAsync(&d_n_out_weights, sizeof(int)*e_len, s2));
			CUDA_CHECK(cudaMallocAsync(&d_n_in_edges, sizeof(int)*e_len, s3));
			CUDA_CHECK(cudaMallocAsync(&d_n_in_edges, sizeof(int)*e_len, s4));

			cudaStreamSynchronize(s3);
			cudaStreamSynchronize(s4);

			cudaStreamDestroy(s3);
			cudaStreamDestroy(s4);

			// kernel that copies from *_edges to n_*_edges for *_valid indices
			t = (max_n_v) * warpSize;
			tpb = min(1024, t);
			blocks = max(1, (t+tpb-1)/tpb);
			d_helper_copy_from_old_to_new<<<blocks, tpb, 0, s1>>>(e_len, d_out_used, d_out_edges,
					d_out_weights, d_out_offset,
					d_n_out_edges, d_n_out_weights, d_n_out_offset,
					max_n_v, d_ptr, false);
			CUDA_CHECK(cudaGetLastError());

			d_helper_copy_from_old_to_new<<<blocks, tpb, 0, s2>>>(e_len, d_in_used, d_in_edges,
					d_in_weights, d_in_offset,
					d_n_in_edges, d_n_in_weights, d_n_in_offset,
					max_n_v, d_ptr, true);
			CUDA_CHECK(cudaGetLastError());

			cudaStreamSynchronize(s1);
			cudaStreamSynchronize(s2);

			// replace old pointers with new ones
			d_out_offset = d_n_out_offset;
			d_out_edges = d_n_out_edges;
			d_out_weights = d_n_out_weights;
			d_in_offset = d_n_in_offset;
			d_in_edges = d_n_in_edges;
			d_in_weights = d_n_in_weights;
		}
		/*	now there is enough space to insert new edges
		 *
		 *	each insertion should check pre-existing edges
		 *		and add if not pre-existing
		 *		else update weights
		 *
		 */
		// drop invalid insertions
		bool *d_valid_ins_pool;
		CUDA_CHECK(cudaMalloc(&d_valid_ins_pool, sizeof(bool)*n_ins));
		CUDA_CHECK(cudaMemset(d_valid_ins_pool, false, n_ins));
		t = ((n_ins + warpSize - 1)/warpSize)*warpSize;
		tpb = min(1024, t);
		blocks = max(1, (t+tpb-1)/tpb);
		d_helper_drop_invalid_ins<<<blocks, tpb, 0, s1>>>(ins, d_valid_ins_pool, n_ins, max_n_v);
		CUDA_CHECK(cudaGetLastError());

		// kernel for only updating pre-existing edges
		// a warp per insertion
		t = n_ins * warpSize;
		tpb = min(1024, t);
		blocks = max(1, (t+tpb-1)/tpb);
		d_helper_update_existing_edges<<<blocks, tpb, 0, s1>>>(ins, n_ins, d_in_edges,
				d_in_weights, d_in_offset, true, d_valid_ins_pool, d_valid);
		CUDA_CHECK(cudaGetLastError());
		d_helper_update_existing_edges<<<blocks, tpb, 0, s2>>>(ins, n_ins, d_out_edges,
				d_out_weights, d_out_offset, false, d_valid_ins_pool, d_valid);
		CUDA_CHECK(cudaGetLastError());

		// reserve slots for new edges
		int *d_out_slots, *d_in_slots;
		CUDA_CHECK(cudaMalloc(&d_out_slots, sizeof(int)*n_ins));
		CUDA_CHECK(cudaMalloc(&d_in_slots, sizeof(int)*n_ins));
		t = n_ins;
		tpb = min(1024, t);
		blocks = max(1, (t+tpb-1)/tpb);

		cudaStreamSynchronize(s1);
		cudaStreamSynchronize(s2);

		d_helper_reserve_slots<<<blocks, tpb, 0, s1>>>(ins, n_ins, 
						d_valid_ins_pool, d_out_used, d_out_slots, false);;
		CUDA_CHECK(cudaGetLastError());

		d_helper_reserve_slots<<<blocks, tpb, 0, s2>>>(ins, n_ins, 
						d_valid_ins_pool, d_in_used, d_in_slots, true);;
		CUDA_CHECK(cudaGetLastError());

		// use the slots to write weights and edges
		// should record the min and max vertex
		// validates any previously removed vertex if inserted again
		// update n_v, n_e
		int *d_n_max_v, *d_n_min_v;
		CUDA_CHECK(cudaMalloc(&d_n_max_v, sizeof(int)));
		CUDA_CHECK(cudaMemset(d_n_max_v, max_v, 1));
		CUDA_CHECK(cudaMalloc(&d_n_min_v, sizeof(int)));
		CUDA_CHECK(cudaMemset(d_n_min_v, 0, 1));

		t = n_ins;
		tpb = min(1024, t);
		blocks = max(1, (t+tpb-1)/tpb);

		cudaStreamSynchronize(s1);
		cudaStreamSynchronize(s2);

		d_helper_insert_new_edges<<<blocks, tpb, 0, s1>>>(ins, n_ins,
							d_out_offset, d_out_edges, d_out_weights,
							d_valid_ins_pool, false, d_out_slots,
							d_valid, d_n_max_v, d_n_min_v, d_ptr);
		CUDA_CHECK(cudaGetLastError());
		
		d_helper_insert_new_edges<<<blocks, tpb, 0, s2>>>(ins, n_ins,
							d_in_offset, d_in_edges, d_in_weights,
							d_valid_ins_pool, true, d_in_slots,
							d_valid, d_n_max_v, d_n_min_v, d_ptr);
		CUDA_CHECK(cudaGetLastError());

		int n_max_v, n_min_v, added_v;
		CUDA_CHECK(cudaMemcpy(&n_max_v, d_n_max_v, sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(&n_min_v, d_n_min_v, sizeof(int), cudaMemcpyDeviceToHost));

		// copy n_e and n_v from device to host
		CUDA_CHECK(cudaMemcpy(&n_v, d_n_v, sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(&n_e, d_n_e, sizeof(int), cudaMemcpyDeviceToHost));
		max_v = max(max_v, n_max_v);
		min_v = min(min_v, n_min_v);

		cudaStreamSynchronize(s1);
		cudaStreamSynchronize(s2);
	}
	cudaStreamDestroy(s1);
	cudaStreamDestroy(s2);
}

CSRView* CSR::to_gpu(){
	// copies object to gpu as CSRView and
	// returns device pointer
	// returns 0 if something fails
	return 0;
}

CSRView* CSR::view(){
	return d_ptr;
}

void CSR::free_gpu(){
	// frees d_ptr
}

template<class O, class F, class P>
int CSR::vmfb(O* op, int n_op, F&& f, bool* flags, int n_f,
			  P&& pred, int* buf, int n_buf){
	// 3-steps
	// applies f on operands in parallel and alters flags based on f
	// synchronization
	// fills into buf based on flags and pred in parallel
}
