#ifndef PMT_H
#define PMT_H
#include <cuda_runtime.h>
#include "graph/types.cuh"

struct BoolStats { int count; int min_idx; int max_idx; };
/*
 * flags should be allocated in device
 */
int       count_true(const bool* flags, int len, cudaStream_t s = 0);

BoolStats bool_stats(const bool* flags, int len, cudaStream_t s = 0);

// ins: device array. pool[i] is true on entry for candidate insertions.
// On exit pool[i] is true only for the last occurrence of each distinct (u,v).
void dedup_insertions(const Edge* ins, bool* pool, int n, cudaStream_t s = 0);

// exclusive scan
void exclusive_scan_inplace(int* arr, int n, cudaStream_t s = 0);

// sum
int sum_int(const int* arr, int len, cudaStream_t s = 0);

#endif
