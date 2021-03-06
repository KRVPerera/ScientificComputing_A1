//
// Created by krv on 12/16/16.
//
#include <cuda.h>
#include<iostream>
#include <getopt.h>
#include <unistd.h>
//#include <random>
#include <algorithm>
#include <omp.h>
#include <A1Config.h>
#include <cuda_runtime.h>
#include <A1Config.h>

using namespace std;

#define MAX_N 1800
#define p_i(x) printf("%d\n",x);

#define HANDLE_ERROR(err) ( HandleError( err, __FILE__, __LINE__))

#define GET_TIME(x);    if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
                { perror("clock_gettime( ):"); exit(EXIT_FAILURE); }

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        cerr << cudaGetErrorString(err) << " in " << file << " at line "  << line << endl;
        exit(EXIT_FAILURE);
    }
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


// inspired by the Cuda Best C programming guide section 9
// and lecture slide 2 TILE_WIDTH == BLOCK_SIZE

#ifdef USE_DOUBLE
__global__ void
matrixMultiKernel(double *C, double *A, double *B, int Width){
    const int BLOCK_SIZE = 16; // NOTE: This must be similar to line 338
    // block indexes
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // thread indexes
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // int col = bx * TILE_WIDTH  + tx
    // int row = by * TILE_WIDTH  + ty

    // Dividing the matrices into sub sections
    // Dividing the matrix A
    int a_begin = Width * BLOCK_SIZE * by;
    int a_end   = a_begin + Width - 1;
    int a_step = BLOCK_SIZE;

    // Dividing the matrix B
    int b_begin = BLOCK_SIZE * bx;
    int b_step = BLOCK_SIZE * Width;

    double temp_c = 0;


    // loop throught the submatrices
    for(int a = a_begin, b = b_begin; a <= a_end;
            a += a_step, b += b_step) {
        // sub matrices
        __shared__ double sub_a[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double sub_b[BLOCK_SIZE][BLOCK_SIZE];

        sub_a[ty][tx] = A[a + Width * ty + tx];
        sub_b[ty][tx] = A[b + Width * ty + tx];

        __syncthreads();


        // loop unroll may not work on cuda if compilation level -O3
        // effects cuda code as wll in the assignment
        // sub matrix multiplication
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            temp_c += sub_a[ty][k] * sub_b[k][tx];
        }
        // sync all the global threads running the computations
        __syncthreads();
    }
    int c = Width * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + Width * ty + tx ] = temp_c;
//    printf("kernel Done \n");
}
#else

__global__ void matrixMultiKernel(float *C, float *A, float *B, int Width) {

    const int BLOCK_SIZE = 16; // NOTE: This must be similar to line 338
    // block indexes
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // thread indexes
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // int col = bx * TILE_WIDTH  + tx
    // int row = by * TILE_WIDTH  + ty

    // Dividing the matrices into sub sections
    // Dividing the matrix A
    int a_begin = Width * BLOCK_SIZE * by;
    int a_end = a_begin + Width - 1;
    int a_step = BLOCK_SIZE;

    // Dividing the matrix B
    int b_begin = BLOCK_SIZE * bx;
    int b_step = BLOCK_SIZE * Width;

    float temp_c = 0;

    // loop throught the submatrices
    for (int a = a_begin, b = b_begin; a <= a_end;
         a += a_step, b += b_step) {
        // sub matrices
        __shared__ float sub_a[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float sub_b[BLOCK_SIZE][BLOCK_SIZE];

        sub_a[ty][tx] = A[a + Width * ty + tx];
        sub_b[ty][tx] = A[b + Width * ty + tx];

        __syncthreads();


        // loop unroll may not work on cuda if compilation level -O3
        // effects cuda code as wll in the assignment
        // sub matrix multiplication
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            temp_c += sub_a[ty][k] * sub_b[k][tx];
        }
        // sync all the global threads running the computations
        __syncthreads();
    }
    int c = Width * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + Width * ty + tx] = temp_c;
//    printf("kernel Done \n");
}

#endif

int main(int argc, char **argv) {
    ios_base::sync_with_stdio(0);
    bool seq_ver, p_ver, cuda_ver, veri_run;
    int c, num_threads = 2;
    struct timespec t0, t1;
    unsigned long sec, nsec;
    unsigned long run_time;
    int N = 600;
    opterr = 1;
    seq_ver = p_ver = cuda_ver = veri_run = false;
    while ((c = getopt(argc, argv, "scp:vn:")) != -1) {
        switch (c) {
            case 'p':
                p_ver = true;
                try {
                    num_threads = atoi(optarg);
                } catch (std::logic_error) {
                    cerr << "Invalid value for -p, set to 8" << endl;
                    num_threads = 2;
                }
                break;
            case 'n':
                try {
                    N = atoi(optarg);
                } catch (std::logic_error) {
                    cerr << "Invalid value for -n, set to 1000" << endl;
                    N = 1000;
                }
                break;
            case 's':
                seq_ver = true;
                break;
            case 'c':
                cuda_ver = true;
                break;
            case 'v':
                veri_run = true;
                break;
            case '?':
                if (optopt == 'p') {
                    cerr << "Option -p requires number of threads" << endl;
                } else {
                    cerr << "Unknown option character" << endl;
                }
                return 1;
            default:
                abort();
        }
    }
    if (num_threads > MAX_THREADS) {
        cerr << "Thread count cannot exceed " << MAX_THREADS << endl;
        abort();
    }

    if (N > MAX_N) {
        cerr << "Please choose a smaller size for N. N should be less than " << MAX_N << endl;
    }
    srand(time(NULL));

#ifdef USE_DOUBLE
    cout << "Running DOUBLE version" << endl;
    cout << "Generating double Matrices of size " << N << "x" << N << "\n";

    double *mat1 = new double [N*N];
    double *mat2 = new double [N*N];
    double *mat_ans ;
    double *mat_p_ans;
    double *mat_c_ans ;
cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    // cuda device pinters
    double *d_mat1,*d_mat2, *d_mat_c_ans;

    mat_c_ans = (double *)malloc(N*N*sizeof(double));
#pragma omp parallel
    {
        double val1;
        double val2;
#pragma omp for schedule (static)
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                 val1 = 1.0*rand()/RAND_MAX + 1;
                 val2 = 1.0*rand()/RAND_MAX + 1;
                mat1[j*N + i] = val1;
                mat2[j*N + i] = val2;

            }
        }
    }
#else
    cout << "Running FLOAT version" << endl;

    cout << "Generating float Matrices of size " << N << "x" << N << "\n";

    float *mat1 = new float[N * N];
    float *mat2 = new float[N * N];
    float *mat_ans;
    float *mat_p_ans;
    float * mat_c_ans = (float *)malloc(N*N*sizeof(float));
//    mat_c_ans = (float *)malloc(N*N*sizeof(float));
    // cuda device pinters
    float *d_mat1, *d_mat2, *d_mat_c_ans;

#pragma omp parallel
    {
        float val1;
        float val2;
#pragma omp for schedule (static)
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                val1 = 1.0 * rand() / RAND_MAX + 1;
                val2 = 1.0 * rand() / RAND_MAX + 1;
                mat1[j * N + i] = val1;
                mat2[j * N + i] = val2;

            }
        }
    }

#endif

    cout << "Matrices creation Done... " << endl;
    if (seq_ver || veri_run) {
        cout << "S >>> Sequential Version running...\n";
#ifdef USE_DOUBLE
        mat_ans = new double [N*N];
#else
        mat_ans = new float[N * N];
#endif
        GET_TIME(t0);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                // mat_ans is already set to zero
                mat_ans[i * N + j] = 0;
                for (int k = 0; k < N; ++k) {
                    mat_ans[i * N + j] += mat1[i * N + k] * mat2[k * N + j];
                }
            }
        }GET_TIME(t1);
        run_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
        cout << "S >>> Sequential Version Elapsed-time(ms) = " << run_time << " ms\n";
    }


    if (p_ver) {
#ifdef USE_DOUBLE
        mat_p_ans = new double [N*N];
#else
        mat_p_ans = new float[N*N];
#endif
        cout << "P >>> Parallel Version running...\n";
        cout << "P >>> number of threads : " << num_threads << "\n";
        //opm
        GET_TIME(t0);
        //    omp_set_num_threads(num_threads);
        double loc_sum;
        int i, j, k;

#pragma omp parallel num_threads(num_threads) private(i,j,k) shared(mat1, mat2,mat_p_ans)
        {
#pragma omp for schedule (static) reduction(+:loc_sum)
            for (i = 0; i < N; ++i) {
                for (j = 0; j < N; ++j) {
                    loc_sum = 0;
                    for (k = 0; k < N; ++k) {
                        loc_sum = loc_sum + (mat1[i * N + k] * mat2[k * N + j]);
                    }
                    mat_p_ans[i * N + j] = loc_sum;
                }
            }
        }GET_TIME(t1);
        run_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
        cout << "P >>> Parallel Version Elapsed-time(ms) = " << run_time << " ms\n";

    }

    if (cuda_ver) {
        cout << "C >>> Cuda version is running...\n";
        int block_size = 16;
        dim3 threads(block_size, block_size);
        dim3 grid(N / block_size, N / block_size);GET_TIME(t0);
#ifdef USE_DOUBLE

        //allocating memory on the device
        HANDLE_ERROR(cudaMalloc((void **) &d_mat1, N * N * sizeof(double)));
        HANDLE_ERROR(cudaMalloc((void **) &d_mat2, N * N *  sizeof(double)));
        HANDLE_ERROR(cudaMalloc((void **) &d_mat_c_ans, N * N * sizeof(double)));
        // copy host memory to device
        HANDLE_ERROR(cudaMemcpy(d_mat1, mat1, N * N * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_mat2, mat2, N * N * sizeof(double), cudaMemcpyHostToDevice));
        matrixMultiKernel<<< grid, threads >>>(d_mat_c_ans, d_mat1, d_mat2, N);
        HANDLE_ERROR(cudaMemcpy(mat_c_ans, d_mat_c_ans, N * N * sizeof(double), cudaMemcpyDeviceToHost));
#else

        HANDLE_ERROR(cudaMalloc((void **) &d_mat1, N * N * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void **) &d_mat2, N * N * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void **) &d_mat_c_ans, N * N * sizeof(float)));
        // copy host memory to device
        HANDLE_ERROR(cudaMemcpy(d_mat1, mat1, N * N * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_mat2, mat2, N * N * sizeof(float), cudaMemcpyHostToDevice));
        matrixMultiKernel << < grid, threads >> > (d_mat_c_ans, d_mat1, d_mat2, N);
        HANDLE_ERROR(cudaMemcpy(mat_c_ans, d_mat_c_ans, N * N * sizeof(float), cudaMemcpyDeviceToHost));
#endif
        GET_TIME(t1);
        run_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
        cout << "S >>> Cuda Version Elapsed-time(ms) = " << run_time << " ms\n";


    }
    bool fail = false;

    if (veri_run) {
#ifdef USE_DOUBLE
        double dot_length = N;
        if (cuda_ver) {
            for (int i = 0; i < N*N; ++i) {
                double abs_err = fabs(mat_c_ans[i] - mat_ans[i]);
                double abs_val = fabs(mat_c_ans[i]);
                double rel_err = abs_err/abs_val/dot_length;
                if (rel_err> MACHINE_ZERO) {
                    fail =  true;
                    cout << "Error! index : " << i << " rel_err : " << rel_err << "  is > " <<  MACHINE_ZERO << endl;
                    break;
                }
            }
        } else if (p_ver) {
            for (int i = 0; i < N*N; ++i) {
                double abs_err = fabs(mat_p_ans[i] - mat_ans[i]);
                double abs_val = fabs(mat_p_ans[i]);
                double rel_err = abs_err/abs_val/dot_length;
                if (rel_err> MACHINE_ZERO) {
                    fail =  true;
                    cout << "Error! index : " << i << " rel_err : " << rel_err << "  is > " <<  MACHINE_ZERO << endl;
                    break;
                }
            }
        }
#else
        float dot_length = N;
        if (cuda_ver) {
            for (int i = 0; i < N * N; ++i) {
                float abs_err = fabs(mat_c_ans[i] - mat_ans[i]);
                float abs_val = fabs(mat_c_ans[i]);
                float rel_err = abs_err / abs_val / dot_length;
                if (rel_err > MACHINE_ZERO) {
                    fail = true;
                    cout << "Error! index : " << i << " rel_err : " << rel_err << "  is > " << MACHINE_ZERO << endl;
                    break;
                }
            }
        } else if (p_ver) {
            for (int i = 0; i < N * N; ++i) {
                float abs_err = fabs(mat_p_ans[i] - mat_ans[i]);
                float abs_val = fabs(mat_p_ans[i]);
                float rel_err = abs_err / abs_val / dot_length;
                if (rel_err > MACHINE_ZERO) {
                    fail = true;
                    cout << "Error! index : " << i << " rel_err : " << rel_err << "  is > " << MACHINE_ZERO << endl;
                    break;
                }
            }
        }
#endif
        if (fail) {
            cout << "Accuracy failed, there are values with relative error > " << MACHINE_ZERO << endl;
        } else {
            cout << "Accuracy OK, there are no values with relative error > " << MACHINE_ZERO << endl;
        }

    }

    // Cleaning up

    delete[] mat1;
    delete[] mat2;

    if(seq_ver || veri_run)
    {
        delete[] mat_ans;
    }

    if (cuda_ver) {
        free(mat_c_ans);
        cudaFree(d_mat_c_ans);
        cudaFree(d_mat1);
        cudaFree(d_mat2);
    }

    if (p_ver) {
        delete[] mat_p_ans;
    }

    std::cout << "Q3 Successfully ran\n";
    return 0;
}
