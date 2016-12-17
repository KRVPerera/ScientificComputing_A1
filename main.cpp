
#include <iostream>
#include <getopt.h>
#include <unistd.h>
#include <random>
#include <algorithm>
#include "Util.h"
#include <omp.h>

using namespace std;


int main(int argc, char **argv) {
    // Program states
    bool seq_ver, p_ver, cuda_ver, veri_run;
    int c, num_threads;
    struct timespec t0, t1;
    float comp_time;
    unsigned long sec, nsec;
    int N = 1000;
    opterr = 1;
    seq_ver = p_ver = cuda_ver = veri_run = false;


    while ((c = getopt(argc, argv, "scp:vn:")) != -1) {
        switch (c) {
            case 'p':
                p_ver = true;
                try {
                    num_threads = stoi(optarg);
                } catch (std::logic_error) {
                    cerr << "Invalid value for -p, set to 8" << endl;
                    num_threads = 8;
                }
                break;
            case 'n':
                try {
                    N = stoi(optarg);
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
    srand(time(NULL));

#ifdef USE_DOUBLE
    double answer = 0;
    double answer_c = 0;
    double answer_p = 0;
    vector<double> local_sum(num_threads);
    cout << "Generating double vectors " << endl;
    vector<double> vector1(N);
    vector<double> vector2(N);
    for (int j = 0; j < N; ++j) {
        double val = random();
        vector1[j] = val;
        val = random();
        vector2[j] = val;
    }
#else
    float answer = 0;
    float answer_c = 0;
    float answer_p = 0;
    cout << "Generating float vectors " << endl;
    vector<float> vector1(N);
    vector<float> vector2(N);
    vector<float> local_sum(num_threads);
    for (int j = 0; j < N; ++j) {
        float val = random();
        vector1[j] = val;
        val = random();
        vector2[j] = val;
    }
#endif

#ifdef USE_DOUBLE
    cout << "Defined : USE_DOUBLE" << endl;
#endif


    if (p_ver) {
        cout << "Parallel Version running" << endl;
        cout << "number of threads : " << num_threads << endl;
        double start_time, run_time;

        start_time = omp_get_wtime();
        #pragma omp parallel
        {
            double x;
            int i, id = omp_get_thread_num();
            num_threads = omp_get_num_threads();
            int istart = (id * N) / num_threads;
            int iend = ((id + 1) * N) / num_threads;

            local_sum[id] = 0;
            for (i = istart; i < iend; i++) {
                local_sum[id] += vector1[i] * vector2[i];
            }
        }

#ifdef USE_DOUBLE
        for(double n : local_sum) {
            answer_p += n;
        }
#else
        for (float n : local_sum) {
            answer_p += n;
        }
#endif
        run_time = omp_get_wtime() - start_time;
        cout << "Parallel Version Elapsed-time(ms) = " << run_time << " ms" << endl;
    }

    if (cuda_ver) {
        cout << "Cuda version is running" << endl;
        answer_c = 0;
    }

    if (seq_ver || veri_run) {
        cout << "Sequential Version running" << endl;GET_TIME(t0);
        for (int i = 0; i < N; ++i) {
            answer += vector1[i] * vector2[i];
        }GET_TIME(t1);
        comp_time = Util::elapsed_time_msec(&t0, &t1, &sec, &nsec);
        cout << "Sequential Version Elapsed-time(ms) = " << comp_time << " ms" << endl;
    }


    if (veri_run) {
        if (cuda_ver) {
            if (fabs(answer - answer_c) > 0.01) {
                cerr << "Values are different" << endl;
            }
        } else if (p_ver) {
            if (fabs(answer - answer_p) > 0.01) {
                cerr << "Values are different" << endl;
            }
        }
    }

    std::cout << "Successful!" << std::endl;
    return 0;

}