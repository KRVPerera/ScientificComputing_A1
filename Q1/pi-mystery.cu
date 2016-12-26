// source: http://cacs.usc.edu/education/cs596/src/cuda/pi.cu

// Using CUDA device to calculate pi Ori

#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <errno.h>
#include <A1Config.h>

#define GET_TIME(x);    if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
                                { perror("clock_gettime( ):"); exit(EXIT_FAILURE); }

float elapsed_time_msec(struct timespec *, struct timespec *, unsigned long *, unsigned long *);


#define NBIN TRIALS_PER_THREAD*BLOCKS*THREADS // 4096*256*256_
#define NUM_BLOCK  BLOCKS  // Number of thread blocks
#define NUM_THREAD  THREADS  // Number of threads per block

#define PI 3.1415926535  // known value of pi
int tid;

#ifdef USE_DOUBLE
double pi_gpu_mys = 0;
//double pi_gpu_cur = 0;

// Kernel that executes on the CUDA device
__global__ void cal_pi(double *sum, int nbin, double step, int nthreads,
                       int nblocks) {
    int i;
    double x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Sequential thread index across the blocks
    for (i = idx; i < nbin; i += nthreads * nblocks) {
        x = (i + 0.5) * step;
        sum[idx] += 4.0 / (1.0 + x * x);
    }
}

#else
float pi_gpu_mys = 0;
//float pi_gpu_cur = 0;

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

#endif

// Main routine that executes on the host
int main(void) {
    struct timespec t0, t1;
    unsigned long sec, nsec;
    float comp_time;
    printf("# NBIN = %d, # of blocks = %d, # of threads/block = %d.\n", NBIN, NUM_BLOCK, NUM_THREAD);

#ifdef USE_DOUBLE
    printf("Running DOUBLE Version \n");
    double *sumHost, *sumDev; // pi-mystery
    double step = 1.0 / (double)NBIN;  // Step size
    size_t size = NUM_BLOCK * NUM_THREAD * sizeof(double);  //Array memory size
#else
    printf("Running FLOAT Version \n");
    float *sumHost, *sumDev; // pi-mystery
    float step = 1.0 / NBIN;  // Step size
    size_t size = NUM_BLOCK * NUM_THREAD * sizeof(float);  //Array memory size
#endif
//    curandState *devStates;

    /// ********* PI Calculation using pi-mystery algorithm
    GET_TIME(t0);
    dim3 dimGrid(NUM_BLOCK, 1, 1);  // Grid dimensions
    dim3 dimBlock(NUM_THREAD, 1, 1);  // Block dimensions

#ifdef USE_DOUBLE
    sumHost = (double *) malloc(size);  //  Allocate array on host
#else
    sumHost = (float *) malloc(size);  //  Allocate array on host
#endif
    cudaMalloc((void **) &sumDev, size);  // Allocate array on device
    // Initialize array in device to 0
    cudaMemset(sumDev, 0, size);
    // Do calculation on device
    cal_pi << < dimGrid, dimBlock >> > (sumDev, NBIN, step, NUM_THREAD, NUM_BLOCK); // call CUDA kernel
    // Retrieve result from device and store it in host array
    cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);

    for (tid = 0; tid < NUM_THREAD * NUM_BLOCK; tid++)
        pi_gpu_mys += sumHost[tid];
    pi_gpu_mys *= step;

    GET_TIME(t1);
    /// ********* PI Calculation using pi-mystery algorithm END
    comp_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
    printf("GPU py-mystery pi calculated in %f s.\n", comp_time);
    printf("CUDA py-mystery estimate of PI = %f [error of %f]\n", pi_gpu_mys, pi_gpu_mys - PI);

    free(sumHost);
    cudaFree(sumDev);

    return 0;
}

float elapsed_time_msec(struct timespec *begin, struct timespec *end, unsigned long *sec, unsigned long *nsec) {
    if (end->tv_nsec < begin->tv_nsec) {
        *nsec = 1000000000 - (begin->tv_nsec - end->tv_nsec);
        *sec = end->tv_sec - begin->tv_sec - 1;
    } else {
        *nsec = end->tv_nsec - begin->tv_nsec;
        *sec = end->tv_sec - begin->tv_sec;
    }
    return (float) (*sec) * 1000 + ((float) (*nsec)) / 1000000;

}
