#include <climits>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include "graph/primitives.cuh"

typedef unsigned long long u64;

/* exclusive scan */

void exclusive_scan_inplace(int* arr, int n, cudaStream_t s){
    if (n <= 0) return;
    thrust::exclusive_scan(thrust::cuda::par.on(s), arr, arr + n, arr);
}

/* reductions */

namespace prim {

struct ToStats {
    const bool* f;
    __host__ __device__ BoolStats operator()(int i) const {
        return f[i] ? BoolStats{1, i, i} : BoolStats{0, INT_MAX, -1};
    }
};

struct MergeStats {
    __host__ __device__ BoolStats operator()(const BoolStats& a,
                                             const BoolStats& b) const {
        return BoolStats{a.count + b.count,
                         a.min_idx < b.min_idx ? a.min_idx : b.min_idx,
                         a.max_idx > b.max_idx ? a.max_idx : b.max_idx};
    }
};

} // namespace prim

int count_true(const bool* flags, int len, cudaStream_t s){
    if (len <= 0) return 0;
    return (int)thrust::count(thrust::cuda::par.on(s), flags, flags + len, true);
}

BoolStats bool_stats(const bool* flags, int len, cudaStream_t s){
    if (len <= 0) return BoolStats{0, -1, -1};
    thrust::counting_iterator<int> first(0);
    BoolStats r = thrust::transform_reduce(
        thrust::cuda::par.on(s), first, first + len,
        prim::ToStats{flags}, BoolStats{0, INT_MAX, -1}, prim::MergeStats{});
    if (r.count == 0){
        r.min_idx = -1;
        r.max_idx = -1;
    }
    return r;
}

/* dedup */

namespace prim {

static constexpr u64 INF = ULLONG_MAX;   // sorts after every real key

// key = (u << 32) | v for candidate insertions, INF for dropped ones
struct MakeKey {
    const Edge* ins;
    const bool* pool;
    __host__ __device__ u64 operator()(int i) const {
        if (!pool[i]) return INF;
        return ((u64)(unsigned)ins[i].u << 32) | (u64)(unsigned)ins[i].v;
    }
};

// in sorted order: keep the last entry of each run of equal keys
struct IsKept {
    const u64* keys;
    int n;
    __host__ __device__ bool operator()(int p) const {
        return keys[p] != INF && (p == n - 1 || keys[p] != keys[p + 1]);
    }
};

} // namespace prim

void dedup_insertions(const Edge* ins, bool* pool, int n, cudaStream_t s){
    if (n <= 0) return;
    auto pol = thrust::cuda::par.on(s);
    thrust::counting_iterator<int> first(0);

    thrust::device_vector<u64>  keys(n);
    thrust::device_vector<int>  idx(n);
    thrust::device_vector<bool> keep(n);

    // 64-bit key per insertion
    thrust::transform(pol, first, first + n, keys.begin(),
                      prim::MakeKey{ins, pool});
    // each entry's original position
    thrust::sequence(pol, idx.begin(), idx.end());
    // stable sort: equal keys become adjacent, in original batch order
    thrust::stable_sort_by_key(pol, keys.begin(), keys.end(), idx.begin());
    // latest of each identical {u,v} edge is valid
    thrust::transform(pol, first, first + n, keep.begin(),
                      prim::IsKept{thrust::raw_pointer_cast(keys.data()), n});
    // write pool's values back to original positions
    thrust::scatter(pol, keep.begin(), keep.end(), idx.begin(), pool);
}

int sum_int(const int* arr, int len, cudaStream_t s){
	if (len <= 0) return 0;
	return thrust::reduce(thrust::cuda::par.on(s), arr, arr + len, 0);
}
