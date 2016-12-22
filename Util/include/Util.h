//
// Created by krv on 12/12/16.
//

#ifndef CS4552_A1_UTIL_H
#define CS4552_A1_UTIL_H

#include <time.h>
#include <vector>

#define GET_TIME(x);	if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
				{ perror("clock_gettime( ):"); exit(EXIT_FAILURE); }

class Util {
public:
//    float static Elapsed_time_msec(timespec *begin, timespec *end,unsigned long *sec,unsigned long *nsec);
//    float static elapsed_time_sec(timespec *begin, timespec *end,unsigned long *sec,unsigned long *nsec);
//
//    static long RequiredSampleSize(float sd, float mean);
//
//    static float Mean(std::vector<float, std::allocator<float>> times);
//
//    static float elapsed_time_nsec(timespec *begin, timespec *end, unsigned long *sec, unsigned long *nsec);
//
//    static float StandardDeviation(std::vector<float> times);
//
//    static float elapsed_time_microsec(timespec *begin, timespec *end, unsigned long *sec, unsigned long *nsec);

    static float elapsed_time_msec(timespec *begin, timespec *end, unsigned long *sec, unsigned long *nsec);
};


#endif //CS4552_A1_UTIL_H
