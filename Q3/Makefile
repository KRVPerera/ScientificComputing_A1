objects = main.o

all: $(objects)
	nvcc -arch=sm_20 -Xcompiler -fopenmp -std=c++11 $(objects) -o Q3_run
%.o: %.cpp
	nvcc -x cu -Xcompiler -fopenmp -arch=sm_20 -std=c++11 -I. -dc $< -o $@
clean:
	rm -f *.o Q3_run
