objects = main.o

all: $(objects)
	nvcc -gencode arch=compute_20,code=sm_20 -Xcompiler -fopenmp -std=c++11 $(objects) -o Q3_run
%.o: %.cpp
	nvcc -x cu -gencode arch=compute_20,code=sm_20 -Xcompiler -fopenmp -std=c++11 -I. -dc $< -o $@
clean:
	rm -f *.o Q3_run
