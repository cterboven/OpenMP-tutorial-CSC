//===----------------------------------------------------------------------===//
//
// Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#define DUMP_ARRAY 0
#define OFFLOAD    1


// declare saxpy_dev() coming from mixed_saxpy_omp_main.cc
void saxpy_hip(size_t, float, float *, float *);
void saxpy_dev(size_t, float, float *, float *);
void saxpy_hst(size_t, float, float *, float *);


void init_array(float * array, int sz, float value) {
    for (int i = 0; i < sz; i++) {
        array[i] = value;
    }
}


void dump_array(float * array, int sz) {
    for (int i = 0; i < sz; i++) {
        printf("%2.1f ", array[i]);
        if (!((i+1) % 16)) {
            printf("\n");
        }
    }
    printf("\n");
}


int main(int argc, char * argv[]) {
    const size_t count = 256*1024*1024;

    float a = 2.0;
    float * x;
    float * y;

    int ndevs;

    x = (float *) malloc(sizeof(float) * count);
    init_array(x, count, -0.5f);
    y = (float *) malloc(sizeof(float) * count);
    init_array(y, count, 2.0f);

    float x0 = x[0];
    float y0 = y[0];

    // allocate the device memory using OpenMP construct
        printf("line %d: [host]   x: %p, y: %p\n", __LINE__, x, y);
#if OFFLOAD
        uintptr_t px = 0;
        uintptr_t py = 0;

        // do pointer translation using OpenMP to get HIP device pointer
            printf("line %d: [host]   x: %p, y: %p\n", __LINE__, x, y);
            saxpy_hip(count, a, x, y); // this call is expected to fail w/o OpenMP directives
#else
        saxpy_hst(count, a, x, y);
#endif

#if DUMP_ARRAY
    dump_array(y, count);
#endif
    double sum = 0.0;
    int cnt = 0;
    for (size_t i = 0; i < count; ++i) {
        sum += y[i];
    }
    sum /= count;
    printf("checksum: %lf (should be %lf)\n", sum, (a * x0 + y0));

    return EXIT_SUCCESS;
}
