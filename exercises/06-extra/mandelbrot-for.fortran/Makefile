BIN_DIR=../bin.fortran
PROGRAM=$(BIN_DIR)/mandelbrot.fortran

CC=gcc
CFLAGS=-O3 -std=gnu99 -fopenmp
FC=gfortran
FFLAGS=-O3 -fopenmp
LDFLAGS=-lm -fopenmp
#VERBOSE=-DVERBOSE

all: $(PROGRAM) 

$(PROGRAM): mandelbrot.f90 lodepng.o lodepng_wrapper.o
	$(FC) $(FFLAGS) $(VERBOSE) -o $@ $^ $(LDFLAGS)

lodepng.o:lib/lodepng.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(BIN_DIR): 
	mkdir $@

clean:
	rm -rf $(PROGRAM) *.o

wipe: clean
	rm -rf *.out *.err
