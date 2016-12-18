
#include <iostream>
#include <getopt.h>
#include <unistd.h>
#include <random>
#include <algorithm>
#include <omp.h>
#include "Util.h"
#include "A1Config.h"

using namespace std;


int main(int argc, char **argv) {
    // Program states
    bool seq_ver, p_ver, cuda_ver, veri_run;
    int c, num_threads = 2;
    struct timespec t0, t1;
    float comp_time;
    unsigned long sec, nsec;
    long N = 100000000;
    opterr = 1;
    seq_ver = p_ver = cuda_ver = veri_run = false;
    ios_base::sync_with_stdio(0);

    while ((c = getopt(argc, argv, "scp:vn:")) != -1) {
        switch (c) {
            case 'p':
                p_ver = true;
                try {
                    num_threads = stoi(optarg);
                } catch (std::logic_error) {
                    cerr << "Invalid value for -p, set to 8" << endl;
                    num_threads = 2;
                }
                break;
            case 'n':
                try {
                    N = stol(optarg);
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
    long double answer_c = 0;
    long double answer_p = 0;
    cout << "Generating double vectors of size " << N << "\n";

//    double vector1[N];
//    double vector2[N];
    double local_sum[num_threads] = {};
    vector<double> vector1(N);
    vector<double> vector2(N);
//    vector<double> local_sum(num_threads);
    double tmp_val;
    for (int j = 0; j < N; ++j) {
        tmp_val = random() % 2 + 1;
        vector1[j] = tmp_val;
        tmp_val = random() % 2 + 1;
        vector2[j] = tmp_val;
    }
#else
    float answer = 0;
    float answer_c = 0;
    float answer_p = 0;
    cout << "Generating float vectors of size " << N << "\n";

    vector<float> vector1(N);
    vector<float> vector2(N);
    float local_sum[num_threads] = {};

    for (int j = 0; j < N; ++j) {
        float val = random()%2+1;
        vector1[j] = val;
        val = random();
        vector2[j] = val;
    }
#endif

#ifdef USE_DOUBLE
    cout << "Defined : USE_DOUBLE" << endl;
#endif
    cout << "Vector creation done " << endl;

    if (p_ver) {
        cout << "P >>> Parallel Version running...\n";
        cout << "P >>> number of threads : " << num_threads << "\n";
        double start_time, run_time;
        //opm
        start_time = omp_get_wtime();
        //int id, istart, iend,i;
        //  #pragma omp parallel shared(local_sum, vector1, vector2, num_threads) private(id, istart, iend, i)
        omp_set_num_threads(num_threads);

        #pragma omp parallel num_threads(num_threads)
        {
            int id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            int istart = floor((id * N) / num_threads);
            int iend = floor(((id + 1) * N) / num_threads);
            if(id == num_threads - 1){
                iend = N;
            }
            local_sum[id] = 0;
            //TODO : Float version error
            for (int i = istart; i < iend; i++) {
                local_sum[id] = local_sum[id] + (vector1[i] * vector2[i]);
            }
        }

        for (int valid = 0; valid < num_threads; valid++) {
            answer_p += local_sum[valid];
        }

        run_time = omp_get_wtime() - start_time;    // Getting the end time for parallel version
        cout << "P >>> Parallel Version Elapsed-time(ms) = " << run_time << " ms\n";
    }

    if (cuda_ver) {
        cout << "C >>> Cuda version is running...\n";
        answer_c = 0;
    }

    if (seq_ver || veri_run) {
        cout << "S >>> Sequential Version running...\n";
        answer = 0;GET_TIME(t0);
        for (int g = 0; g < N; ++g) {
            answer += (vector1[g] * vector2[g]);
        }GET_TIME(t1);

        comp_time = Util::elapsed_time_msec(&t0, &t1, &sec, &nsec);
        cout << "S >>> Sequential Version Elapsed-time(ms) = " << comp_time << " ms\n";
    }


    if (veri_run) {
        if (cuda_ver) {
            if (fabs(answer - answer_c) > 0.01f) {
                cout << "Values are different" << endl;
                cout << "C >>> Cuda Version Answer: " << answer_c << "\n";
            }
        } else if (p_ver) {
            if (fabs(answer - answer_p) > 0.01f) {
                cout << "Values are different" << endl;
                cout << "P >>> Parallel Version Answer: " << answer_p << "\n";
            }
        }
        cout << "S >>> Serial Version Answer: " << answer << "\n";
        cout << "Diff : " << fabs(answer - answer_p) << "\n";
    }

    std::cout << "Successful! \n";
    return 0;

}