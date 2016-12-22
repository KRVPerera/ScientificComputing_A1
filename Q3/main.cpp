//
// Created by krv on 12/16/16.
//
#include <cuda.h>
#include<iostream>
#include <getopt.h>
#include <unistd.h>
#include <random>
#include <algorithm>
#include <omp.h>
#include "A1Config.h"

using namespace std;

#define MAX_N 1800

#define GET_TIME(x);	if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
				{ perror("clock_gettime( ):"); exit(EXIT_FAILURE); }

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


int main(int argc, char **argv) {
    ios_base::sync_with_stdio(0);
    bool seq_ver, p_ver, cuda_ver, veri_run;
    int c, num_threads = 2;
    struct timespec t0, t1;
    unsigned long sec, nsec;
    unsigned long  run_time;
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
            case 'g':
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
    cout << "Generating double Matrices of size " << N << "x" << N << "\n";

    double **mat1 = new double *[N];
    double **mat2 = new double *[N];
    double **mat_ans = new double *[N];
    double **mat_p_ans = new double *[N];
//    double local_sum[num_threads] = {};

#pragma omp parallel
    {

#pragma omp for schedule (static)
        for (int j = 0; j < N; ++j) {
            mat1[j] = new double[N];
            mat2[j] = new double[N];
            mat_ans[j] = new double[N];
            mat_p_ans[j] = new double[N];
            for (int i = 0; i < N; ++i) {
                double val1 = 1.0*random()/RAND_MAX + 1;
                double val2 = 1.0*random()/RAND_MAX + 1;
                mat1[j][i] = val1;
                mat2[j][i] = val2;
                mat_ans[j][i] = 0;
                mat_p_ans[j][i] = 0;

            }
        }
    }
#else

    cout << "Generating float Matrices of size " << N << "x" << N << "\n";

    float **mat1 = new float *[N];
    float **mat2 = new float *[N];
    float **mat_ans = new float *[N];
    float **mat_p_ans = new float *[N];
    omp_set_num_threads(num_threads);

#pragma omp parallel
    {
#pragma omp for schedule (static)
        for (int j = 0; j < N; ++j) {
            mat1[j] = new float[N];
            mat2[j] = new float[N];
            mat_ans[j] = new float[N];
            mat_p_ans[j] = new float[N];
            for (int i = 0; i < N; ++i) {
                float val1 = 1.0*random()/RAND_MAX + 1;
                float val2 = 1.0*random()/RAND_MAX + 1;
                mat1[j][i] = val1;
                mat2[j][i] = val2;
                mat_ans[j][i] = 0;
                mat_p_ans[j][i] = 0;
            }
        }
    }

#endif

#ifdef USE_DOUBLE
    cout << "Defined : USE_DOUBLE" << endl;
#endif
    cout << "Matrices creation Done... " << endl;

    if (p_ver) {
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
                        loc_sum = loc_sum + (mat1[i][k] * mat2[k][j]);
                    }
                    mat_p_ans[i][j] = loc_sum;
                }
            }
        }
        GET_TIME(t1);
        run_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
        cout << "P >>> Parallel Version Elapsed-time(ms) = " << run_time << " ms\n";
    }

    if (cuda_ver) {
        cout << "C >>> Cuda version is running...\n";
        GET_TIME(t0);

        GET_TIME(t1);
        run_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
        cout << "S >>> Cuda Version Elapsed-time(ms) = " << run_time << " ms\n";
    }

    if (seq_ver || veri_run) {
        cout << "S >>> Sequential Version running...\n";

        GET_TIME(t0);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                // mat_ans is already set to zero
                for (int k = 0; k < N; ++k) {
                    mat_ans[i][j] += mat1[i][k] * mat2[k][j];
                }
            }
        }
        GET_TIME(t1);
        run_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
        cout << "S >>> Sequential Version Elapsed-time(ms) = " << run_time << " ms\n";
    }


    if (veri_run) {
        double diff = 0;
        if (cuda_ver) {

        } else if (p_ver) {

            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    diff += fabs(mat_p_ans[i][j] - mat_ans[i][j]);
                }
            }
        }
        cout << "Diff : " << diff << "\n";
    }

    // Cleaning up
    for (int l = N; l > 0;) {
        delete[] mat1[--l];
        delete[] mat2[l];
        delete[] mat_ans[l];
    }
    delete[] mat1;
    delete[] mat2;
    delete[] mat_ans;

    std::cout << "Q3 Successfully ran\n";
    return 0;
}