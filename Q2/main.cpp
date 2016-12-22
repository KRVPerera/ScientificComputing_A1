#include <cuda.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <inttypes.h>
#include <omp.h>
#include <host_defines.h>
#include <device_launch_parameters.h>
#include "A1Config.h"

#define MAX_THREADS 20
#define pi(x) printf("%d\n",x);

__global__ void dotPro(int n, double *vec1, double *vec2, double *vec3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec3[i] = vec1[i] * vec2[i];
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

#define GET_TIME(x);	if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
				{ perror("clock_gettime( ):"); exit(EXIT_FAILURE); }

//#define USE_DOUBLE
int main(int argc, char **argv) {
    // Program states
    int seq_ver, p_ver, cuda_ver, veri_run;
    int c, num_threads = 2;
    struct timespec t0, t1;
    float comp_time;
    unsigned long sec, nsec;
    long N = 10000;
    opterr = 1;
    seq_ver = p_ver = cuda_ver = veri_run = 0;

    while ((c = getopt(argc, argv, "scp:vn:")) != -1) {
        switch (c) {
            case 'p':
                p_ver = 1;
                num_threads = atoi(optarg);
                if (num_threads == 0) {
                    fprintf(stderr, "Invalid value for -p, set to 8\n");
                    num_threads = 2;
                }
                break;
            case 'n':
                N = atoi(optarg);
                if (N > 1000000000 || N <= 0) {
                    fprintf(stderr, "Invalid value for -n, set to 1000\n");
                    N = 1000;
                }
                break;
            case 's':
                seq_ver = 1;
                break;
            case 'c':
                cuda_ver = 1;
                break;
            case 'v':
                veri_run = 1;
                break;
            case '?':
                if (optopt == 'p') {
                    printf("Option -p requires number of threads\n");
                } else {
                    printf("Unknown option character\n");
                }
                return 1;
            default:
                abort();
        }
    }
    if (num_threads > MAX_THREADS) {
        printf("Thread count cannot exceed %d\n", MAX_THREADS);
        abort();
    }

    srand(time(NULL));

#ifdef USE_DOUBLE
    long double answer = 0;
    long double *answer_c = (long double *) malloc(sizeof(long double));
    *answer_c = 0.0;
    long double answer_p = 0;
    printf("Generating double vectors of size  %ld\n", N);

    double *h_vector1 = (double *) malloc(sizeof(double) * N);
    double *h_vector2 = (double *) malloc(sizeof(double) * N);
    double *h_vector3 = (double *) malloc(sizeof(double) * N);
    double * d_vector1, * d_vector2, *d_vector3;

    double tmp_val;
    for (int j = 0; j < N; ++j) {
        tmp_val = 1.0 * rand() / RAND_MAX + 1;
        h_vector1[j] = tmp_val;
        tmp_val = 1.0 * rand() / RAND_MAX + 1;
        h_vector2[j] = tmp_val;
    }

    printf("Vector 2 first %f\n", h_vector2[0]);
#else
    float answer = 0;
    float answer_c = 0;
    float answer_p = 0;

    printf("Generating float vectors of size  %d\n",N);

    float * h_vector1 = (float * ) malloc(sizeof(float)*N);
    float * h_vector2 = (float * ) malloc(sizeof(float)*N);
    float * h_vector3 = (float * ) malloc(sizeof(float)*N);
    float * d_vector1, * d_vector2, *d_vector3;

#pragma omp parallel num_threads(num_threads)
    {
#pragma omp for
        for (int j = 0; j < N; ++j) {
            float val = 1.0*random()/RAND_MAX + 1;
            h_vector1[j] = val;
            val = 1.0*random()/RAND_MAX + 1;
            h_vector2[j] = val;
        }
    }

#endif

#ifdef USE_DOUBLE
    printf("Defined : USE_DOUBLE\n");
#endif
    printf("Vector creation done\n");

    if (p_ver) {
        printf("P >>> Parallel Version running...\n");
        printf("P >>> number of threads : %d\n", num_threads);
        //  #pragma omp parallel shared(local_sum, h_vector1, h_vector2, num_threads) private(id, istart, iend, i)
        GET_TIME(t0);

//      Manual parallelalisation
//        #pragma omp parallel num_threads(num_threads)
//        {
//            int id = omp_get_thread_num();
//            int num_threads = omp_get_num_threads();
//            int istart = floor((id * N) / num_threads);
//            int iend = floor(((id + 1) * N) / num_threads);
//            if(id == num_threads - 1){
//                iend = N;
//            }
//            local_sum[id] = 0;
//
//            for (int i = istart; i < iend; i++) {
//                local_sum[id] = local_sum[id] + (h_vector1[i] * h_vector2[i]);
//            }
//        }
//        for (int valid = 0; valid < num_threads; valid++) {
//            answer_p += local_sum[valid];
//        }

#pragma omp parallel num_threads(num_threads)
        {
#pragma omp for schedule(static) reduction(+:answer_p)
            for (int i = 0; i < N; i++) {
                answer_p = answer_p + (h_vector1[i] * h_vector2[i]);
            }
        }

        GET_TIME(t1);    // Getting the end time for parallel version
        comp_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
        printf("P >>> Parallel Version Elapsed-time(ms) = %lf ms\n", comp_time);
    }

    if (seq_ver || veri_run) {
        printf("S >>> Sequential Version running...\n");
        answer = 0.0;GET_TIME(t0);
        for (int g = 0; g < N; ++g) {
            answer += (h_vector1[g] * h_vector2[g]);

        }GET_TIME(t1);

        comp_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
        printf("S >>> Sequential Version Elapsed-time(ms) = %lf ms\n", comp_time);
    }



    if (cuda_ver) {
        printf("C >>> Cuda version is running...\n")GET_TIME(t0);
        int th_p_block = 256;
        int blocks = (N+255)/th_p_block;
        GET_TIME(t0);
        cudaMalloc((void **) &d_vector1, N * sizeof(double));
        cudaMalloc((void **) &d_vector2, N * sizeof(double));
        cudaMalloc((void **) &d_vector3, N * sizeof(double));
	cudaError_t err =  cudaMemcpy(d_vector1, h_vector1, N * sizeof(double), cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
		printf("Cuda memocopy not success...\n");
		abort();
	}
        cudaMemcpy(d_vector2, h_vector2, N * sizeof(double), cudaMemcpyHostToDevice);

        dotPro <<<blocks, th_p_block>>> (N, d_vector1, d_vector2, d_vector3);


        cudaMemcpy(h_vector3, d_vector3, N * sizeof(double), cudaMemcpyDeviceToHost);
        *answer_c = 0;
        for (int i = 0; i < N; ++i) {
            *answer_c += h_vector3[i];
        }

        GET_TIME(t1);
        comp_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
        printf("P >>> Cuda Version Elapsed-time(ms) = %lf ms\n", comp_time);
    }

        if (veri_run) {
        if (cuda_ver) {
            if (fabs(answer - *answer_c) > 0.01) {
                printf("Values are different\n");
                printf("C >>> Cuda Version Answer: %Lf\n", *answer_c);
            }
        } else if (p_ver) {
            if (fabs(answer - answer_p) > 0.01) {
                printf("Values are different\n");
                printf("P >>> Parallel Version Answer: %Lf\n", answer_p);
            }
        }
        printf("Values are different\n");
        printf("S >>> Serial Version Answer: %Lf\n", answer);
        printf("Diff : %Lf\n", fabs(answer - answer_p));
    }

    free(h_vector1);
    cudaFree(d_vector1);
    free(h_vector2);
    cudaFree(d_vector2);
    free(h_vector3);
    cudaFree(d_vector3);
    free(answer_c);
    printf("Q2 Successful ran..! \n");
    return 0;

}
