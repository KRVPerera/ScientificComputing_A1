objects = main.o

all: $(objects)
	nvcc -gencode arch=compute_20,code=sm_20 -Xcompiler -fopenmp  $(objects) -o Q2_run

%.o: %.cpp
	nvcc -x cu -gencode arch=compute_20,code=sm_20 -Xcompiler -fopenmp -I. -dc $< -o $@

clean:
	rm -f *.o Q2_run