//===----------------------------------------------------------------------===//
//
// Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <assert.h>

#include "hip/hip_runtime.h"

#define HIPCALL(func)                                                \
    {                                                                \
        hipError_t ret = func;                                       \
        if (ret != hipSuccess) {                                     \
            fprintf(stderr,                                          \
                    "HIP error: '%s' at %s:%d\n",                    \
                    hipGetErrorString(ret), __FUNCTION__, __LINE__); \
            abort();                                                 \
        }                                                            \
    }

__global__ void saxpy(size_t n, float a, float * x, float * y) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

void saxpy_hip(size_t n, float a, float * x, float * y) {
    const int tbsz = 256;
    saxpy<<<(n+tbsz-1)/tbsz,tbsz,0,NULL>>>(n, a, x, y);
    HIPCALL(hipStreamSynchronize(nullptr));
}

void saxpy_dev(size_t n, float a, float * x, float * y) {
    printf("Offloading to GPU via OpenMP\n");
    #pragma omp target teams distribute parallel for \
                schedule(nonmonotonic:static,1)
    for (size_t i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

void saxpy_hst(size_t n, float a, float * x, float * y) {
    #pragma omp parallel for firstprivate(a)
    for (size_t i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}
