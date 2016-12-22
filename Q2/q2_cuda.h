#ifndef CS4552_A1_Q2_CUDA_H
#define CS4552_A1_Q2_CUDA_H

#include <cuda.h>
#include <host_defines.h>


__global__ void dotPro(int n, float *vec1, float *vec2, float *vec3);

float Run(int n, double *vec1, double *vec2, double *vec3);

#endif //CS4552_A1_Q2_CUDA_H