// Source: http://web.mit.edu/pocky/www/cudaworkshop/MonteCarlo/PiMyRandom.cu

// Written by Barry Wilkinson, UNC-Charlotte. PiMyRandom.cu  December 22, 2010.
//Derived somewhat from code developed by Patrick Rogers, UNC-C

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <A1Config.h>

#define PI 3.1415926535  // known value of pi

float elapsed_time_msec(struct timespec *begin, struct timespec *end,
                        unsigned long *sec, unsigned long *nsec);


#define GET_TIME(x);    if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
                { perror("clock_gettime( ):"); exit(EXIT_FAILURE); }

#ifdef USE_DOUBLE
__device__ double my_rand(unsigned int *seed) {
	unsigned long a = 16807;  // constants for random number generator
	unsigned long m = 2147483647;   // 2^31 - 1
	unsigned long x = (unsigned long) *seed;

	x = (a * x)%m;

	*seed = (unsigned int) x;

        return ((double)x)/m;
}

__global__ void gpu_monte_carlo(double *estimate) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	double x, y;

	unsigned int seed =  tid + 1;  // starting number in random sequence

	for(int i = 0; i < TRIALS_PER_THREAD; i++) {
		x = my_rand(&seed);
		y = my_rand(&seed);
		points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
	}
	estimate[tid] = 4.0f * points_in_circle / (double) TRIALS_PER_THREAD; // return estimate of pi
}

float host_monte_carlo(long trials) {
	float x, y;
	long points_in_circle;
	for(long i = 0; i < trials; i++) {
		x = rand() / (double) RAND_MAX;
		y = rand() / (double) RAND_MAX;
		points_in_circle += (x*x + y*y <= 1.0f);
	}
	return 4.0f * points_in_circle / (double) trials;
}
#else
__device__ float my_rand(unsigned int *seed) {
	unsigned long a = 16807;  // constants for random number generator
	unsigned long m = 2147483647;   // 2^31 - 1
	unsigned long x = (unsigned long) *seed;

	x = (a * x)%m;

	*seed = (unsigned int) x;

        return ((float)x)/m;
}

__global__ void gpu_monte_carlo(float *estimate) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	float x, y;

	unsigned int seed =  tid + 1;  // starting number in random sequence

	for(int i = 0; i < TRIALS_PER_THREAD; i++) {
		x = my_rand(&seed);
		y = my_rand(&seed);
		points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
	}
	estimate[tid] = 4.0f * points_in_circle / (float) TRIALS_PER_THREAD; // return estimate of pi
}

float host_monte_carlo(long trials) {
	float x, y;
	long points_in_circle;
	for(long i = 0; i < trials; i++) {
		x = rand() / (float) RAND_MAX;
		y = rand() / (float) RAND_MAX;
		points_in_circle += (x*x + y*y <= 1.0f);
	}
	return 4.0f * points_in_circle / trials;
}

#endif
int main (int argc, char *argv[]) {
    struct timespec t0, t1;
    float comp_time;
    unsigned long sec, nsec;

    #ifdef USE_DOUBLE
    printf("Running DOUBLE version\n");
	double host[BLOCKS * THREADS];
	double *dev;
	double pi_cpu;
	double pi_gpu;
    #else
	printf("Running FLOAT version\n");
	float host[BLOCKS * THREADS];
	float *dev;
	float pi_cpu;
	float pi_gpu;
#endif


	printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n", TRIALS_PER_THREAD,
BLOCKS, THREADS);

    GET_TIME(t0);
#ifdef USE_DOUBLE
	cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(double)); // allocate device mem. for counts

	gpu_monte_carlo<<<BLOCKS, THREADS>>>(dev);

	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(double), cudaMemcpyDeviceToHost); // return results
#else
	cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(float)); // allocate device mem. for counts

	gpu_monte_carlo<<<BLOCKS, THREADS>>>(dev);

	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost); // return results 
#endif
	pi_gpu = 0;
	for(int i = 0; i < BLOCKS * THREADS; i++) {
		pi_gpu += host[i];
	}

	pi_gpu /= (BLOCKS * THREADS);

    GET_TIME(t1);

    comp_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
    printf("GPU pi calculated in \t%9.3f s.\n", comp_time);

    GET_TIME(t0);
	pi_cpu = host_monte_carlo(BLOCKS * THREADS * TRIALS_PER_THREAD);
    GET_TIME(t1);
    comp_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
	printf("CPU pi calculated in \t%9.3f s.\n", comp_time);

	printf("CUDA estimate of PI \t= %f [error of %f]\n", pi_gpu, pi_gpu - PI);
	printf("CPU estimate of PI \t= %f [error of %f]\n", pi_cpu, pi_cpu - PI);
	
	return 0;
}


float elapsed_time_msec(struct timespec *begin, struct timespec *end,
                        unsigned long *sec, unsigned long *nsec) {
    if (end->tv_nsec < begin->tv_nsec) {
        *nsec = 1000000000 - (begin->tv_nsec - end->tv_nsec);
        *sec = end->tv_sec - begin->tv_sec - 1;
    } else {
        *nsec = end->tv_nsec - begin->tv_nsec;
        *sec = end->tv_sec - begin->tv_sec;
    }
    return (float) (*sec) * 1000 + ((float) (*nsec)) / 1000000.0;
}