#include <iostream>
#include <getopt.h>
#include <unistd.h>
//#include <pthread.h>
#include <random>
#include "Util.h"

using namespace std;

// vector size
#ifndef N
#define N 1000
#endif
//#define DBL_PREC

int main(int argc, char **argv) {
    // Program states
    pthread_mutex_t mutex_t;
    pthread_mutex_init(&mutex_t, NULL);
    bool seq_ver, p_ver, cuda_ver, veri_run;
    int c, num_threads;
    struct timespec t0, t1;
    float comp_time;
    unsigned long 		sec, nsec;
    opterr = 1;
    seq_ver = p_ver = cuda_ver = veri_run = false;

    while ((c = getopt(argc, argv, "scp:v")) != -1) {
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

#ifdef DBL_PREC
    double answer = 0;
    double answer_s = 0;
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
    float answer_s = 0;
    cout << "Generating float vectors " << endl;
    vector<float> vector1(N);
    vector<float> vector2(N);
    for (int j = 0; j < N; ++j) {
        float val = random();
        vector1[j] = val;
        val = random();
        vector2[j] = val;
    }
#endif




    if (p_ver) {
        cout << "Parallel Version running" << endl;
        cout << "number of threads : " << num_threads << endl;
        pthread_t threadpool[num_threads];
        answer_s = 0;
    }

    if (cuda_ver) {
        cout << "Cuda version is running" << endl;
        answer_s = 0;
    }

    if (seq_ver || veri_run) {
        cout << "Sequential Version running" << endl;
        GET_TIME(t0);
        for (int i = 0; i < N; ++i) {
            answer += vector1[i] * vector2[i];
        }
        GET_TIME(t1);
        comp_time = Util::Elapsed_time_msec(&t0, &t1, &sec, &nsec);
        cout << "Sequential Version Elapsed-time(ms) = " << comp_time << " ms" << endl;
    }


    if(veri_run){
        if(fabs(answer-answer_s) > 0.01){
            cerr << "Values are different" << endl;
        }
    }

    std::cout << "Hello, World!" << std::endl;
    return 0;


}