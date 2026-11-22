#include <cuda.h>
#include "graph/exclusive_scan.cuh"
#include "graph/checks.cuh"

__device__ __forceinline__ int warp_inclusive_scan(int v, int lane){
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1){
        int t = __shfl_up_sync(0xFFFFFFFF, v, d);
        if (lane >= d) v += t;
    }
    return v;
}

// Pass 1: every block scans its 1024-element tile (exclusive, in place)
// and writes the tile total to block_sums (skipped if block_sums == nullptr).
__global__ void scan_tiles(int* arr, int n, int* block_sums){
    __shared__ int warp_off[32];
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;
    int i    = blockIdx.x * SCAN_BLOCK + tid;

    int x   = (i < n) ? arr[i] : 0;
    int inc = warp_inclusive_scan(x, lane);

    if (lane == 31) warp_off[wid] = inc;          // one write per warp
    __syncthreads();

    if (wid == 0){                                // scan the 32 warp totals
        int w    = warp_off[lane];
        int winc = warp_inclusive_scan(w, lane);
        warp_off[lane] = winc - w;                // exclusive warp offsets
        if (lane == 31 && block_sums) block_sums[blockIdx.x] = winc;
    }
    __syncthreads();

    if (i < n) arr[i] = inc - x + warp_off[wid];  // exclusive result
}

// Pass 3: add each block's scanned offset to its tile.
__global__ void add_block_offsets(int* arr, int n, const int* block_offsets){
    int i = blockIdx.x * SCAN_BLOCK + threadIdx.x;
    if (blockIdx.x > 0 && i < n) arr[i] += block_offsets[blockIdx.x];
}

// arr[i] = sum of arr[0..i-1], in place, arr is a device pointer
void exclusive_scan_inplace(int* arr, int n, cudaStream_t s = 0){
    if (n <= 0) return;
    int blocks = (n + SCAN_BLOCK - 1) / SCAN_BLOCK;
    if (blocks == 1){
        scan_tiles<<<1, SCAN_BLOCK, 0, s>>>(arr, n, nullptr);
        return;
    }
    int* block_sums;
    CUDA_CHECK(cudaMalloc(&block_sums, sizeof(int) * blocks));
    scan_tiles<<<blocks, SCAN_BLOCK, 0, s>>>(arr, n, block_sums);
    exclusive_scan_inplace(block_sums, blocks, s);      // recurse on the sums
    add_block_offsets<<<blocks, SCAN_BLOCK, 1, s>>>(arr, n, block_sums);
    CUDA_CHECK(cudaFree(block_sums));                   // implicit sync, see note
}
