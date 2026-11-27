#include <cuda_runtime.h>
#include "common/batch_upload.h"
#include "graph/checks.cuh"

template <class T>
static T* to_device(const std::vector<T>& v){
	if (v.empty()) return nullptr;
	T* d;
	CUDA_CHECK(cudaMalloc((void**)&d, sizeof(T) * v.size()));
	CUDA_CHECK(cudaMemcpy(d, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
	return d;
}

Batch upload_batch(const HostBatch& h){
	Batch b;
	b.n_ins = (int)h.ins.size();
	b.n_del = (int)h.del.size();
	b.n_rem = (int)h.rem.size();
	b.ins = to_device(h.ins);
	b.del = to_device(h.del);
	b.v   = to_device(h.rem);
	return b;
}

void free_batch(Batch& b){
	cudaFree(b.ins);
	cudaFree(b.del);
	cudaFree(b.v);
	b.ins = nullptr;
	b.del = nullptr;
	b.v = nullptr;
	b.n_ins = b.n_del = b.n_rem = 0;
}
