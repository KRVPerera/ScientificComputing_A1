// source: http://cacs.usc.edu/education/cs596/src/cuda/pi.cu

// Using CUDA device to calculate pi Ori

#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <errno.h>


#define GET_TIME(x);	if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
								{ perror("clock_gettime( ):"); exit(EXIT_FAILURE); }

float 	elapsed_time_msec(struct timespec *, struct timespec *, long *, long *);

#define TRIALS_PER_THREAD 4096
#define BLOCKS 256
#define THREADS 256
#define NBIN 268435456 // 4096*256*256_
#define NUM_BLOCK  30  // Number of thread blocks
#define NUM_THREAD  8  // Number of threads per block

#define PI 3.1415926535  // known value of pi
int tid;
float pi_gpu_mys = 0;
float pi_gpu_cur = 0;

// Kernel that executes on the CUDA device
__global__ void cal_pi(float *sum, int nbin, float step, int nthreads,
	int nblocks) {
	int i;
	float x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x; // Sequential thread index across the blocks
	for (i = idx; i < nbin; i += nthreads * nblocks) {
		x = (i + 0.5) * step;
		sum[idx] += 4.0 / (1.0 + x * x);
	}
}


__global__ void gpu_monte_carlo(float *estimate, curandState *states) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	float x, y;

	curand_init(1234, tid, 0, &states[tid]);  // 	Initialize CURAND

	for (int i = 0; i < TRIALS_PER_THREAD; i++) {
		x = curand_uniform(&states[tid]);
		y = curand_uniform(&states[tid]);
		points_in_circle += (x * x + y * y <= 1.0f); // count if x & y is in the circle.
	}
	estimate[tid] = 4.0f * points_in_circle / (float)TRIALS_PER_THREAD; // return estimate of pi
}
// Main routine that executes on the host
int main(void) {
	clock_t start, stop;
	struct timespec 	t0, t1;
	unsigned long 		sec, nsec;
	float 			comp_time;
	printf(
		"# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n",
		TRIALS_PER_THREAD,
		NUM_BLOCK, NUM_THREAD);

	float *sumHost, *sumDev; // pi-mystery
	float *monteHost, *monteDev; // pi-curand
	float *dev;
	curandState *devStates;
	float host[BLOCKS * THREADS];

	/// ********* PI Calculation using pi-mystery algorithm
	start = clock();
	dim3 dimGrid(NUM_BLOCK, 1, 1);  // Grid dimensions
	dim3 dimBlock(NUM_THREAD, 1, 1);  // Block dimensions
	float step = 1.0 / NBIN;  // Step size
	size_t size = NUM_BLOCK * NUM_THREAD * sizeof(float);  //Array memory size
	sumHost = (float *)malloc(size);  //  Allocate array on host
	cudaMalloc((void **)&sumDev, size);  // Allocate array on device
	// Initialize array in device to 0
	cudaMemset(sumDev, 0, size);
	// Do calculation on device
	cal_pi << <dimGrid, dimBlock >> >(sumDev, NBIN, step, NUM_THREAD, NUM_BLOCK); // call CUDA kernel
	// Retrieve result from device and store it in host array
	cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);

	for (tid = 0; tid < NUM_THREAD * NUM_BLOCK; tid++)
		pi_gpu_mys += sumHost[tid];
	pi_gpu_mys *= step;
	stop = clock();
	/// ********* PI Calculation using pi-mystery algorithm END
	printf("GPU py-mystery pi calculated in %f s.\n", (stop - start) / (float)CLOCKS_PER_SEC);


	/// ********* PI Calculation using pi-curand algorithm START
	start = clock();

	cudaMalloc((void **)&dev, BLOCKS * THREADS * sizeof(float)); // allocate device mem. for counts

	cudaMalloc((void **)&devStates, THREADS * BLOCKS * sizeof(curandState));

	gpu_monte_carlo << <BLOCKS, THREADS >> >(dev, devStates);

	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float),
		cudaMemcpyDeviceToHost); // return results

	pi_gpu_cur = 0;
	for (int i = 0; i < BLOCKS * THREADS; i++) {
		pi_gpu_cur += host[i];
	}

	pi_gpu_cur /= (BLOCKS * THREADS);

	stop = clock();

	/// ********* PI Calculation using pi-curand algorithm END
	printf("GPU py-curand pi calculated in %f s.\n", (stop - start) / (float)CLOCKS_PER_SEC);
	printf("CUDA py-mystery estimate of PI = %f [error of %f]\n", pi_gpu_mys, pi_gpu_mys - PI);
	printf("CUDA py-curand estimate of PI = %f [error of %f]\n", pi_gpu_cur, pi_gpu_cur - PI);
	// Print results

	// Cleanup
	free(sumHost);
	cudaFree(sumDev);

	return 0;
}


float elapsed_time_msec(struct timespec *begin, struct timespec *end, long *sec, long *nsec)
{
	if (end->tv_nsec < begin->tv_nsec) {
		*nsec = 1000000000 - (begin->tv_nsec - end->tv_nsec);
		*sec = end->tv_sec - begin->tv_sec - 1;
	}
	else {
		*nsec = end->tv_nsec - begin->tv_nsec;
		*sec = end->tv_sec - begin->tv_sec;
	}
	return (float)(*sec) * 1000 + ((float)(*nsec)) / 1000000;

}
