BIN_DIR=../bin.fortran
PROGRAM=$(BIN_DIR)/jacobi.fortran

FC=gfortran
FFLAGS=-O3 -fopenmp
LDFLAGS=-lm -fopenmp
#VERBOSE=-DVERBOSE

all: $(PROGRAM) 

$(PROGRAM): jacobi.f90
	$(FC) $(FFLAGS) $(VERBOSE) -o $@ $^ $(LDFLAGS)

$(BIN_DIR): 
	mkdir $@

clean:
	rm -rf $(PROGRAM) *.o

wipe: clean
	rm -rf *.out *.err
