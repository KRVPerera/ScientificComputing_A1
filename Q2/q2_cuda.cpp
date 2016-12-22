#include "q2_cuda.h"
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include "Common.h"


__global__ void dotPro(int n, double *vec1, double *vec2, double *vec3)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) vec3[i] = vec1[i]*vec2[i];
}



float Run(int N, double * h_vector3, double * h_vector1, double * h_vector2){
    struct timespec t0, t1;
    unsigned long sec, nsec;
    int th_p_block = 256;
    float comp_time;
    long double * answer_c = (long double *)malloc(sizeof(long double));
    memset(answer_c,0,sizeof(long double));
    int blocks = (N+(th_p_block-1))/th_p_block;
    double * d_vector1, * d_vector2, *d_vector3;

    GET_TIME(t0);

    cudaMalloc((void **)&d_vector1, N*sizeof(double));
    cudaMalloc((void **)&d_vector2, N*sizeof(double));
    cudaMalloc((void **)&d_vector3, N*sizeof(double));
    cudaMemcpy(d_vector1, h_vector1, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector2, h_vector2, N*sizeof(double), cudaMemcpyHostToDevice);

    dotPro<<<blocks,th_p_block>>>(N,d_vector1, d_vector2, d_vector3);

    cudaMemcpy(h_vector3, d_vector3, N*sizeof(double), cudaMemcpyDeviceToHost);
    answer_c = 0;
    for (int i = 0; i < N; ++i) {
        *answer_c += h_vector3[i];
    }
    GET_TIME(t1);
    comp_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
    return comp_time;
}