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
#define th_p_block  256


static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}


#ifdef USE_DOUBLE
__global__ void dotPro(long n, double *vec1, double *vec2, double *vec3) {

    __shared__ double cache[th_p_block];
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int cacheIdx =  threadIdx.x;
	double temp = 0;
	while(tid < n)
	{
		temp += vec1[tid] * vec2[tid];
		tid += blockDim.x * gridDim.x;
	}

	cache[cacheIdx] = temp;
    __syncthreads();

	// reduction
    unsigned i = blockDim.x/2; // need the num threads to be a power of two (256 is okay)
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
#else
__global__ void dotPro(long n, float *vec1, float *vec2, float *vec3) {

    __shared__ float cache[th_p_block];
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int cacheIdx =  threadIdx.x;
    float temp = 0;
    while(tid < n)
    {
        temp += vec1[tid] * vec2[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = temp;
    __syncthreads();

    // reduction
    unsigned i = blockDim.x/2; // need the num threads to be a power of two (256 is okay)
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

#endif

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
    int blocks  = (N+th_p_block-1)/th_p_block;
    printf("Blocks %d\n", blocks);
    srand(time(NULL));

#ifdef USE_DOUBLE
    printf("Running DOUBLE version \n");
    long double answer = 0;
    long double answer_c = 0;// = (long double *) malloc(sizeof(long double));
    long double answer_p = 0;
    printf("Generating double vectors of size  %ld\n", N);

    double * h_vector1 = (double *) malloc(sizeof(double) * N);
    double * h_vector2 = (double *) malloc(sizeof(double) * N);
    double * h_vector3 = (double * ) malloc(sizeof(double)*blocks);
    double * d_vector1, * d_vector2, * d_vector3;

    #pragma omp parallel
    {
        double tmp_val;
        #pragma omp for
        for (int j = 0; j < N; ++j) {
        tmp_val = 1.0 * rand() / RAND_MAX + 1;
        h_vector1[j] = tmp_val;
        tmp_val = 1.0 * rand() / RAND_MAX + 1;
        h_vector2[j] = tmp_val;
        }
    }

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    printf("Vector 2 first %f\n", h_vector2[0]);
#else
    printf("Running float version \n");
    float answer = 0;
    float answer_c = 0;
    float answer_p = 0;

    printf("Generating float vectors of size  %ld\n",N);

    float * h_vector1 = (float * ) malloc(sizeof(float)*N);
    float * h_vector2 = (float * ) malloc(sizeof(float)*N);
   // float * h_vector3;
    float * h_vector3 = (float * ) malloc(sizeof(float)*blocks);
    float * d_vector1, * d_vector2, * d_vector3;

#pragma omp parallel
    {
        float val;
#pragma omp for
        for (int j = 0; j < N; ++j) {
            val = 1.0*rand()/RAND_MAX + 1;
            h_vector1[j] = val;
            val = 1.0*rand()/RAND_MAX + 1;
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
        answer_p = 0;
        #pragma omp parallel num_threads(num_threads) shared(h_vector1, h_vector2)
        {

            #pragma omp for schedule(static) reduction(+:answer_p)
            for (int i = 0; i < N; i++) {
                answer_p = answer_p+ (h_vector1[i] * h_vector2[i]);
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
        h_vector3 = (double *) malloc(sizeof(double)*blocks);
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
        h_vector3 = (float *) malloc(sizeof(float)*blocks);
        HANDLE_ERROR(cudaMalloc((void **) &d_vector1, N * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void **) &d_vector2, N * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void **) &d_vector3, blocks * sizeof(float)));
        // copy host memory to device
        HANDLE_ERROR(cudaMemcpy(d_vector1, h_vector1, N * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_vector2, h_vector2, N * sizeof(float), cudaMemcpyHostToDevice));

        // kerneal code
        dotPro <<< blocks, th_p_block >>> (N, d_vector1, d_vector2, d_vector3);

        // copy device memory back to host memory
        HANDLE_ERROR(cudaMemcpy(h_vector3, d_vector3, blocks * sizeof(float), cudaMemcpyDeviceToHost));
#endif
	       // serial portions summation
        answer_c = 0;
        for (int i = 0; i < blocks; ++i) {
            answer_c += h_vector3[i];
        }

        GET_TIME(t1);
        comp_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
        printf("P >>> Cuda Version Elapsed-time(ms) = %lf ms\n", comp_time);

    }

#ifdef USE_DOUBLE
    if (veri_run) {
        printf("S >>> Serial Version Answer: %Lf\n", answer);

        if (cuda_ver) {
            printf("C >>> Cuda Version Answer: %Lf\n", answer_c);
            double abs_err = fabs(answer_c-answer);
            double dot_length = N;
            double abs_val = fabs(answer_c);
            double rel_err = abs_err/abs_val/dot_length;
            if (rel_err> MACHINE_ZERO) {
                printf("Values are different\n");
                printf("Error! relative error is > %E\n", MACHINE_ZERO);
            }else{
                printf("Values are similar\n");
                printf("Error! rel_err %E relative error is < %E\n",rel_err, MACHINE_ZERO);
            }
            printf("Diff : %f\n", abs_err);
        } else if (p_ver) {
            printf("P >>> Parallel Version Answer: %Lf\n", answer_p);
            double abs_err = fabs(answer_p-answer);
            double dot_length = N;
            double abs_val = fabs(answer_p);
            double rel_err = abs_err/abs_val/dot_length;
            if (rel_err> MACHINE_ZERO) {
                printf("Error! rel_err %E relative error is > %E\n",rel_err, MACHINE_ZERO);
            }else{
                printf("Values are similar\n");
                printf("Error! rel_err %E relative error is < %E\n",rel_err, MACHINE_ZERO);
            }
            printf("Diff : %f\n", abs_err);
        }
    }
#else
    if (veri_run) {
        printf("S >>> Serial Version Answer: %f\n", answer);

        if (cuda_ver) {
            printf("C >>> Cuda Version Answer: %f\n", answer_c);
            float abs_err = fabs(answer_c-answer);
            float dot_length = N;
            float abs_val = fabs(answer_c);
            float rel_err = abs_err/abs_val/dot_length;
            if (rel_err> MACHINE_ZERO) {
                printf("Values are different\n");
                printf("Error! relative error is > %E\n", MACHINE_ZERO);
            }else{
                printf("Values are similar\n");
                printf("Error! rel_err %E relative error is < %E\n",rel_err, MACHINE_ZERO);
            }
            printf("Diff : %f\n", abs_err);
        } else if (p_ver) {
            printf("P >>> Parallel Version Answer: %f\n", answer_p);
            float abs_err = fabs(answer_p-answer);
            float dot_length = N;
            float abs_val = fabs(answer_p);
            float rel_err = abs_err/abs_val/dot_length;
            if (rel_err> MACHINE_ZERO) {
                printf("Error! rel_err %E relative error is > %E\n",rel_err, MACHINE_ZERO);
            }else{
                printf("Values are similar\n");
                printf("Error! rel_err %E relative error is < %E\n",rel_err, MACHINE_ZERO);
            }
            printf("Diff : %f\n", abs_err);

        }
    }
#endif
    free(h_vector1);
    free(h_vector2);
    free(h_vector3);

    if(cuda_ver){
        cudaFree(d_vector1);
        cudaFree(d_vector2);
        cudaFree(d_vector3);
    }

    printf("Q2 Successful run..! \n");
    return 0;

}
