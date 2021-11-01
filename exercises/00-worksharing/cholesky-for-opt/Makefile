BIN_DIR=../bin
PROGRAM=$(BIN_DIR)/cholesky-for-opt

CC=gcc
CFLAGS=-O3 -std=gnu99 -I$(AMDBLIS_INSTALL_ROOT)/include -I$(AMDLIBFLAME_INSTALL_ROOT)/include -fopenmp
LDFLAGS=-L$(AMDBLIS_INSTALL_ROOT)/lib -L$(AMDLIBFLAME_INSTALL_ROOT)/lib -lblis-mt -lflame -llapacke -lm -fopenmp
#VERBOSE=-DVERBOSE

all: $(PROGRAM) 

$(PROGRAM): cholesky.c 
	$(CC) $(CFLAGS) $(VERBOSE) -o $@ $^ $(LDFLAGS)

$(BIN_DIR): 
	mkdir $@

clean:
	rm -rf $(PROGRAM) *.o

wipe: clean
	rm -rf *.out *.err
