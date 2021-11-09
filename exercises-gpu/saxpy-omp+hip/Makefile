.PHONY: all clean

all: hybrid_saxpy_omp_hip

hybrid_saxpy_omp_hip: hybrid_saxpy_omp_main.o hybrid_saxpy_hip_kernel.o
	clang++ -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 \
	        $(shell hipcc --ldflags) -lamdhip64 -o $@ hybrid_saxpy_omp_main.o hybrid_saxpy_hip_kernel.o
hybrid_saxpy_hip_kernel.o: hybrid_saxpy_hip_kernel.cc
	hipcc $(HIPCCFLAGS) -o $@ -c $<
hybrid_saxpy_omp_main.o: hybrid_saxpy_omp_main.cc
	clang++ -O2 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 -o $@ -c $<

clean:
	rm -f *.o *.a *.so
	rm -f hybrid_saxpy_omp_hip
