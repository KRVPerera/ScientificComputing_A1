//
// Created by krv on 12/12/16.
//

#include "../include/Util.h"
#include <cmath>
#include <algorithm>

using namespace std;

long Util::RequiredSampleSize(float sd, float mean) {
    long N = (long) ceil(((float) 100 * 1.960 * sd) / (5 * mean));
    return N;
}

float Util::Mean(vector<float> times) {
    int size = times.size();
    float sum;
    sum = accumulate(times.begin(), times.end(), 0);
    return (float) sum / size;
}

float Util::StandardDeviation(vector<float> times) {
    double mean = Util::Mean(times);
    double variance = 0;
    for (int i = 0; i < times.size(); ++i) {
        variance += pow(times.at(i) - mean, 2);
    }
    variance = variance / times.size();
    return sqrt(variance);
}

float Util::elapsed_time_nsec(struct timespec *begin, struct timespec *end,
                              unsigned long *sec, unsigned long *nsec) {
    if (end->tv_nsec < begin->tv_nsec) {
        *nsec = 1000000000 - (begin->tv_nsec - end->tv_nsec);
        *sec = end->tv_sec - begin->tv_sec - 1;
    } else {
        *nsec = end->tv_nsec - begin->tv_nsec;
        *sec = end->tv_sec - begin->tv_sec;
    }
    return (float) (*sec) * 1000000 + ((float) (*nsec));
}


float Util::elapsed_time_microsec(struct timespec *begin, struct timespec *end,
                                  unsigned long *sec, unsigned long *nsec) {
    if (end->tv_nsec < begin->tv_nsec) {
        *nsec = 1000000000 - (begin->tv_nsec - end->tv_nsec);
        *sec = end->tv_sec - begin->tv_sec - 1;
    } else {
        *nsec = end->tv_nsec - begin->tv_nsec;
        *sec = end->tv_sec - begin->tv_sec;
    }
    return (float) (*sec) * 1000000 + ((float) (*nsec)) / 1000.0;
}

float Util::elapsed_time_msec(struct timespec *begin, struct timespec *end,
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