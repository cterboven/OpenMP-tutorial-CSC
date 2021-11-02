BIN_DIR=../bin.fortran
PROGRAM=$(BIN_DIR)/quicksort.fortran

FC=gfortran
FFLAGS=-c -O3 -fopenmp 
LDFLAGS=-fopenmp

all: $(PROGRAM) 

$(PROGRAM): quicksort.f90
	$(FC) $(FFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.f90
	$(FC) $(FFLAGS) -o $@ $^ 

sudoku.cpp:
	@cat README.md 
	@false

$(BIN_DIR): 
	mkdir $@

clean:
	rm -rf $(PROGRAM) *.o

wipe: clean
	rm -rf *.out *.err
