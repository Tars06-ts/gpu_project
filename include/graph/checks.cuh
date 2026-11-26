#ifndef CHECK_H

#define CHECK_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

// Macro for Runtime API calls (returns cudaError_t)
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t result = (call);                                     \
        if (result != cudaSuccess) {                                     \
            std::fprintf(stderr, "CUDA Runtime Error:\n");               \
            std::fprintf(stderr, "  File: %s\n  Line: %d\n",             \
                         __FILE__, __LINE__);                            \
            std::fprintf(stderr, "  Error: %s (%s)\n",                   \
                         cudaGetErrorName(result),                       \
                         cudaGetErrorString(result));                    \
            std::exit(EXIT_FAILURE);                                     \
        }                                                                \
    } while (0)
#endif

#ifndef FATAL_H
#define FATAL_H

#define FATAL(format, ...) do { \
		fprintf(stderr, "[FATAL] (%s:%d): " format "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
		exit(EXIT_FAILURE); \
	} while(0)

#endif
#endif
