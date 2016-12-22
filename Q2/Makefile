objects = main.o

all: $(objects)
	nvcc -arch=sm_20 $(objects) -o Q2_run

%.o: %.cpp
	nvcc -x cu -arch=sm_20 -I. -dc $< -o $@

clean:
	rm -f *.o Q2_run
