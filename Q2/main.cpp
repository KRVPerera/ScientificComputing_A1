#include <time.h>
#include <errno.h>
#include <iostream>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <algorithm>
#include <inttypes.h>
#include <algorithm>
#include <omp.h>
#include "q2_cuda.h"
#include <A1Config.h>
#include "Common.h"

using namespace std;

//#define USE_DOUBLE
int main(int argc, char **argv) {
    // Program states
    bool seq_ver, p_ver, cuda_ver, veri_run;
    int c, num_threads = 2;
    struct timespec t0, t1;
    float comp_time;
    unsigned long sec, nsec;
    long N = 10000;
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

    srand(time(NULL));

#ifdef USE_DOUBLE
    long double answer = 0;
    long double * answer_c = (long double *)malloc(sizeof(long double));
    long double answer_p = 0;
    cout << "Generating double vectors of size " << N << "\n";

    double * h_vector1 = (double * ) malloc(sizeof(double)*N);
    double * h_vector2 = (double * ) malloc(sizeof(double)*N);
    double * h_vector3 = (double * ) malloc(sizeof(double)*N);
//    double * d_vector1, * d_vector2, *d_vector3;

    double tmp_val;
    for (int j = 0; j < N; ++j) {
        tmp_val = 1.0*rand()/RAND_MAX + 1;
        h_vector1[j] = tmp_val;
        tmp_val = 1.0*rand()/RAND_MAX + 1;
        h_vector2[j] = tmp_val;
    }
#else
    float answer = 0;
    float answer_c = 0;
    float answer_p = 0;
    cout << "Generating float vectors of size " << N << "\n";

    float * h_vector1 = (float * ) malloc(sizeof(float)*N);
    float * h_vector2 = (float * ) malloc(sizeof(float)*N);
    float * d_vector1, * d_vector2, *d_vector3, *h_vector3;

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp for
        for (int j = 0; j < N; ++j) {
            float val = 1.0*random()/RAND_MAX + 1;
            h_vector1[j] = val;
            val = random();
            h_vector2[j] = val;
        }
    }
#endif

#ifdef USE_DOUBLE
    cout << "Defined : USE_DOUBLE" << endl;
#endif
    cout << "Vector creation done " << endl;

    if (p_ver) {
        cout << "P >>> Parallel Version running...\n";
        cout << "P >>> number of threads : " << num_threads << "\n";
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
        cout << "P >>> Parallel Version Elapsed-time(ms) = " << comp_time << " ms\n";
    }

    if (cuda_ver) {
        cout << "C >>> Cuda version is running...\n";
//        int th_p_block = 256;
//        int blocks = (N+(th_p_block-1))/th_p_block;
        GET_TIME(t0);
        Run(N, h_vector3, h_vector1, h_vector2);
        GET_TIME(t1);
        comp_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
        cout << "P >>> Cuda Version Elapsed-time(ms) = " << comp_time << " ms\n";
        answer_c = 0;
    }

    if (seq_ver || veri_run) {
        cout << "S >>> Sequential Version running...\n";
        answer = 0;
        GET_TIME(t0);
        for (int g = 0; g < N; ++g) {
            answer += (h_vector1[g] * h_vector2[g]);
        }
        GET_TIME(t1);

        comp_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
        cout << "S >>> Sequential Version Elapsed-time(ms) = " << comp_time << " ms\n";
    }


    if (veri_run) {
        if (cuda_ver) {
            if (fabs(answer - *answer_c) > 0.1) {
                cout << "Values are different" << endl;
                cout << "C >>> Cuda Version Answer: " << answer_c << "\n";
            }
        } else if (p_ver) {
            if (fabs(answer - answer_p) > 0.1) {
                cout << "Values are different" << endl;
                cout << "P >>> Parallel Version Answer: " << answer_p << "\n";
            }
        }
        cout << "S >>> Serial Version Answer: " << answer << "\n";
        cout << "Diff : " << fabs(answer - answer_p) << "\n";
    }

    free(h_vector1);
    free(h_vector2);
    std::cout << "Q2 Successful ran..! \n";
    return 0;

}
