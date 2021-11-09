all: jacobi.gpu-mi100

jacobi.gpu-mi100: Makefile jacobi.c
	clang $(DEBUG) -O3 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 -lm -o jacobi.gpu-mi100 jacobi.c

jacobi.gpu-mi50: Makefile jacobi.c
	clang $(DEBUG) -O3 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 -lm -o jacobi.gpu-mi100 jacobi.c

clean:
	rm -f *.o
	rm -f jacobi.gpu-mi100
	rm -f jacobi.gpu-mi50

realclean: clean
	rm -f *.bak *~

.PHONY: all clean realclean
