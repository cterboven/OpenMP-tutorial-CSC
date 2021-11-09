# STREAM_ARRAY_SIZE=2500000000
STREAM_ARRAY_SIZE=250000000
# STREAM_ARRAY_SIZE=25000000
# STREAM_ARRAY_SIZE=2500000
# STREAM_ARRAY_SIZE=250000
# STREAM_ARRAY_SIZE=25000

NTIMES=10

DEBUG=-g

TYPES=-DDO_COPY=1 -DDO_ADD=1 -DDO_SCALE=1 -DDO_TRIAD=1

all: stream.gpu-mi100

stream.gpu-mi50: Makefile stream.c
	clang $(DEBUG) -O3 $(TYPES) -DGPU_STREAM=1 -DGPU_DATA_ENV=1 -DSTATIC -DSTREAM_TYPE=double -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) -DNTIMES=$(NTIMES) -mcmodel=large -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 -o stream.gpu-mi50 stream.c

stream.gpu-mi100: Makefile stream.c
	clang $(DEBUG) -O3 $(TYPES) -DGPU_STREAM=1 -DGPU_DATA_ENV=1 -DSTATIC -DSTREAM_TYPE=double -DSTREAM_ARRAY_SIZE=$(STREAM_ARRAY_SIZE) -DNTIMES=$(NTIMES) -mcmodel=large -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 -o stream.gpu-mi100 stream.c

clean:
	rm -f *.o
	rm -f stream.gpu-mi50
	rm -f stream.gpu-mi100

realclean: clean
	rm -f *.bak *~

.PHONY: all clean realclean
