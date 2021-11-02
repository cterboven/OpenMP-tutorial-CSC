BIN_DIR=../bin
PROGRAM=$(BIN_DIR)/sin-cos

FC=gfortran
FFLAGS=-O3 -fopenmp
LDFLAGS=-lm -fopenmp
#VERBOSE=-DVERBOSE

all: $(PROGRAM) 

$(PROGRAM): sin-cos.f90
	$(FC) $(FFLAGS) $(VERBOSE) -o $@ $^ $(LDFLAGS)

$(BIN_DIR): 
	mkdir $@

clean:
	rm -rf $(PROGRAM) *.o

wipe: clean
	rm -rf *.out *.err
