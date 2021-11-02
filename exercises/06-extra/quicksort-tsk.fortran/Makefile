BIN_DIR=../bin.fortran
PROGRAM=$(BIN_DIR)/quicksort.fortran

FC=gfortran
FFLAGS=-O3 -fopenmp 
LDFLAGS=-fopenmp

all: $(PROGRAM) 

$(PROGRAM): quicksort.o
	$(FC) $(FFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.f90
	$(FC) $(FFLAGS) -c -o $@ $^ 

$(BIN_DIR): 
	mkdir $@

clean:
	rm -rf $(PROGRAM) *.o

wipe: clean
	rm -rf *.out *.err
