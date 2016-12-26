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
#include <A1Config.h>

#define MAX_THREADS 20
#define pi(x) printf("%d\n",x);
#define HANDLE_ERROR(err) ( HandleError( err, __FILE__, __LINE__ ) )

#define th_p_block 32*4
#define blocks  (N + th_p_block-1) / th_p_block

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void dotPro(int n, double *vec1, double *vec2, double *vec3) {

    __shared__ double cache[th_p_block];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cacheIdx =  threadIdx.x;
	float temp = 0;
	while(tid < n)
	{
		temp += vec1[tid] * vec2[tid];
		tid += blockDim.x * gridDim.x;
	}

	cache[cacheIdx] = temp;

	// reduction
	int i = blockDim.x/2; // need the num threads to be a power of two (256 is okay)
	while( i != 0 ){
		if(cacheIdx < i){
			cache[cacheIdx] += cache[cacheIdx + i ];
		}

		__syncthreads(); //sync threads in the current block
        // power of two needed here
		i = i/2;
	}
	if(cacheIdx == 0){
		vec3[blockIdx.x] = cache[0];
	}
//    if (tid < n) vec3[i] = vec1[i] * vec2[i];
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

#define GET_TIME(x);    if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
                { perror("clock_gettime( ):"); exit(EXIT_FAILURE); }


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
                if (N > 100000000 || N <= 0) {
                    fprintf(stderr, "Invalid value for -n, set to 100000000\n");
                    N = 100000000;
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
    long double answer_c;// = (long double *) malloc(sizeof(long double));
    long double answer_p = 0;
    printf("Generating double vectors of size  %ld\n", N);

    double *h_vector1 = (double *) malloc(sizeof(double) * N);
    double *h_vector2 = (double *) malloc(sizeof(double) * N);
    double * h_vector3 = (double *) malloc(sizeof(double)*blocks);
    double * d_vector3;
    double *d_vector1, *d_vector2;

    double tmp_val;
    for (int j = 0; j < N; ++j) {
        tmp_val = 1.0 * rand() / RAND_MAX + 1;
        h_vector1[j] = tmp_val;
        tmp_val = 1.0 * rand() / RAND_MAX + 1;
        h_vector2[j] = tmp_val;
    }

   // cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
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
    sleep(1);
    if (p_ver) {
        printf("P >>> Parallel Version running...\n");
        printf("P >>> number of threads : %d\n", num_threads);
        //  #pragma omp parallel shared(local_sum, h_vector1, h_vector2, num_threads) private(id, istart, iend, i)
        GET_TIME(t0);

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
    sleep(1);
    if (seq_ver || veri_run) {
        printf("S >>> Sequential Version running...\n");
        GET_TIME(t0);
        answer = 0.0;

        for (int g = 0; g < N; ++g) {
            answer += (h_vector1[g] * h_vector2[g]);
        }GET_TIME(t1);

        comp_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
        printf("S >>> Sequential Version Elapsed-time(ms) = %lf ms\n", comp_time);
    }

    sleep(1);
    if (cuda_ver) {
        printf("C >>> Cuda version is running...\n")GET_TIME(t0);
        GET_TIME(t0);
        // memory allocation on device
        #ifdef USE_DOUBLE
	      HANDLE_ERROR(cudaMalloc((void **) &d_vector1, N * sizeof(double)));
        HANDLE_ERROR(cudaMalloc((void **) &d_vector2, N * sizeof(double)));
        HANDLE_ERROR(cudaMalloc((void **) &d_vector3, blocks * sizeof(double)));
      	// copy host memory to device
        HANDLE_ERROR(cudaMemcpy(d_vector1, h_vector1, N * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_vector2, h_vector2, N * sizeof(double), cudaMemcpyHostToDevice));

	      // kerneal code
        dotPro <<< blocks, th_p_block >>> (N, d_vector1, d_vector2, d_vector3);

	      // copy device memory back to host memory
        HANDLE_ERROR(cudaMemcpy(h_vector3, d_vector3, blocks * sizeof(double), cudaMemcpyDeviceToHost));
#else
      fprintf(stderr, "Float processing of cuda not yet implemented\n");
#endif
	       // serial portions summation
        answer_c = 0;
        for (int i = 0; i < blocks; ++i) {
            answer_c += h_vector3[i];
        }

        GET_TIME(t1);
        comp_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
        printf("P >>> Cuda Version Elapsed-time(ms) = %lf ms\n", comp_time);
	cudaFree(d_vector3);
	free(h_vector3);
    }

    if (veri_run) {
        printf("S >>> Serial Version Answer: %Lf\n", answer);

        if (cuda_ver) {
            printf("C >>> Cuda Version Answer: %Lf\n", answer_c);
            if (fabs(answer - answer_c) > 0.01) {
                printf("Values are different\n");
            }else{
                printf("Values are similar\n");
            }
            printf("Diff : %Lf\n", fabs(answer - answer_c));
        } else if (p_ver) {
            printf("P >>> Parallel Version Answer: %Lf\n", answer_p);
            if (fabs(answer - answer_p) > 0.01) {
                printf("Values are different\n");
            }else{
                printf("Values are similar\n");
            }
            printf("Diff : %Lf\n", fabs(answer - answer_p));
        }
    }

    free(h_vector1);
    free(h_vector2);
    cudaFree(d_vector1);
    cudaFree(d_vector2);
    printf("Q2 Successful ran..! \n");
    return 0;

}
