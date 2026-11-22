#ifndef EXCLUSIVE_SCAN_H

#include <cuda.h>

#define EXCLUSIVE_SCAN_H
#define SCAN_BLOCK 1024   // 32 warps per block

void exclusive_scan_inplace(int*, int, cudaStream_t);

#endif
