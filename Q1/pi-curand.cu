// Source: http://web.mit.edu/pocky/www/cudaworkshop/MonteCarlo/Pi.cu

// Written by Barry Wilkinson, UNC-Charlotte. Pi.cu  December 22, 2010.
//Derived somewhat from code developed by Patrick Rogers, UNC-C

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <cuda.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <omp.h>
#include <getopt.h>
#include <unistd.h>
#include <A1Config.h>

#define PI 3.1415926535  // known value of pi

float elapsed_time_msec(struct timespec *begin, struct timespec *end,
                        unsigned long *sec, unsigned long *nsec);

#define GET_TIME(x);	if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
				{ perror("clock_gettime( ):"); exit(EXIT_FAILURE); }

#ifdef USE_DOUBLE
__global__ void gpu_monte_carlo(double *estimate, curandState *states) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	double x, y;

	curand_init(1234, tid, 0, &states[tid]);  // 	Initialize CURAND


	for(int i = 0; i < TRIALS_PER_THREAD; i++) {
		x = curand_uniform (&states[tid]);
		y = curand_uniform (&states[tid]);
		points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
	}
	estimate[tid] = 4.0f * points_in_circle / (double) TRIALS_PER_THREAD; // return estimate of pi
}

double host_monte_carlo(long trials) {
	double x, y;
	long points_in_circle = 0;
	for(long i = 0; i < trials; i++) {
		x = rand() / (double) RAND_MAX;
		y = rand() / (double) RAND_MAX;
		points_in_circle += (x*x + y*y <= 1.0f);
	}
	return 4.0f * points_in_circle / (double)trials;
}

double host_monte_carlo_p(long trials, int nthreads) {
    double x, y;
    long points_in_circle=0;
    long tot_trials = trials/nthreads;
    long local_sum = 0;
    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for schedule (static) reduction(+:local_sum)
        for (long i = 0; i < tot_trials; i++) {
            x = rand() / (double) RAND_MAX;
            y = rand() / (double) RAND_MAX;
            local_sum += (x * x + y * y <= 1.0f);
        }

        #pragma omp critical
        points_in_circle += local_sum;
    }

    return 4.0f * points_in_circle / (double)trials;
}
#else

__global__ void gpu_monte_carlo(float *estimate, curandState *states) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	float x, y;

	curand_init(1234, tid, 0, &states[tid]);  // 	Initialize CURAND


	for(int i = 0; i < TRIALS_PER_THREAD; i++) {
		x = curand_uniform (&states[tid]);
		y = curand_uniform (&states[tid]);
		points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
	}
	estimate[tid] = 4.0f * points_in_circle / (float) TRIALS_PER_THREAD; // return estimate of pi
}

float host_monte_carlo(long trials) {
	float x, y;
	long points_in_circle = 0;
	for(long i = 0; i < trials; i++) {
		x = rand() / (float) RAND_MAX;
		y = rand() / (float) RAND_MAX;
		points_in_circle += (x*x + y*y <= 1.0f);
	}
	return 4.0f * points_in_circle / trials;
}

float host_monte_carlo_p(long trials, int nthreads) {
    float x, y;
    long points_in_circle = 0;
    long tot_trials = trials/nthreads;
    long local_sum = 0;
    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for schedule (static) reduction(+:local_sum)
        for (long i = 0; i < tot_trials; i++) {
            x = rand() / (float) RAND_MAX;
            y = rand() / (float) RAND_MAX;
            local_sum += (x * x + y * y <= 1.0f);
        }

        #pragma omp critical
        points_in_circle += local_sum;
    }

    return 4.0f * points_in_circle / trials;
}
#endif



int main (int argc, char *argv[]) {
    struct timespec t0, t1;
    float comp_time;
    unsigned long sec, nsec;
#ifdef USE_DOUBLE
    printf("Running the DOUBLE Version\n");
    double host[BLOCKS * THREADS];
	double *dev;
    double pi_cpu2;
    double pi_gpu;
    double pi_cpu;
#else
    printf("Running the FLOAT Version\n");
	float host[BLOCKS * THREADS];
	float *dev;
    float pi_cpu2;
    float pi_gpu;
    float pi_cpu;
#endif
	curandState *devStates;
    int c, num_threads = 4;
    opterr = 1;
    while ((c = getopt(argc, argv, "hp:")) != -1) {
        switch (c) {
            case 'p':
                num_threads = atoi(optarg);
                if(num_threads == 0) {
                 fprintf(stderr, "Invalid value for -p, set to 4\n");
                    num_threads = 4;
                }
                break;
            case '?':
                if (optopt == 'p') {
                    fprintf(stderr, "Option -p requires number of threads\n");
                } else {
                    fprintf(stderr, "Unknown option character\n");
                }
                return 1;
            default:
                abort();
        }
    }


	printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n", TRIALS_PER_THREAD,
BLOCKS, THREADS);

    GET_TIME(t0);

#ifdef USE_DOUBLE
    cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(double)); // allocate device mem. for counts

	cudaMalloc( (void **)&devStates, THREADS * BLOCKS * sizeof(curandState) );

	gpu_monte_carlo<<<BLOCKS, THREADS>>>(dev, devStates);

	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(double), cudaMemcpyDeviceToHost); // return results


	for(int i = 0; i < BLOCKS * THREADS; i++) {
		pi_gpu += host[i];
	}

	pi_gpu /= (BLOCKS * THREADS);
#else
	cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(float)); // allocate device mem. for counts
	
	cudaMalloc( (void **)&devStates, THREADS * BLOCKS * sizeof(curandState) );

	gpu_monte_carlo<<<BLOCKS, THREADS>>>(dev, devStates);

	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost); // return results


	for(int i = 0; i < BLOCKS * THREADS; i++) {
		pi_gpu += host[i];
	}

	pi_gpu /= (BLOCKS * THREADS);
#endif
    GET_TIME(t1);
    comp_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);

	printf("GPU pi calculated in \t\t%9.3f ms.\n", comp_time);


    GET_TIME(t0);
    pi_cpu2 = host_monte_carlo_p(BLOCKS * THREADS * TRIALS_PER_THREAD, num_threads);
    GET_TIME(t1);
    comp_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
    printf("CPU parellel pi calculated in \t%9.3f ms using %d threads.\n", comp_time, num_threads);


    GET_TIME(t0);
    pi_cpu = host_monte_carlo(BLOCKS * THREADS * TRIALS_PER_THREAD);
    GET_TIME(t1);
    comp_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
	printf("CPU pi calculated in \t\t%9.3f ms.\n", comp_time);


	printf("CUDA estimate of PI \t\t= %f \t[error of %f]\n", pi_gpu, pi_gpu - PI);
	printf("CPU estimate of PI \t\t= %f \t[error of %f]\n", pi_cpu, pi_cpu - PI);
	printf("CPU parallel estimate of PI \t= %f \t[error of %f]\n", pi_cpu2, pi_cpu2 - PI);

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
