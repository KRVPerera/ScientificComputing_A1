// Source: http://docs.nvidia.com/cuda/curand/index.html

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <curand_kernel.h>

#include <iostream>
#include <iomanip>

float elapsed_time_msec(struct timespec *begin, struct timespec *end,
                        unsigned long *sec, unsigned long *nsec);


#define GET_TIME(x);    if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
                { perror("clock_gettime( ):"); exit(EXIT_FAILURE); }


// we could vary M & N to find the perf sweet spot
struct estimate_pi :
        public thrust::unary_function<unsigned int, float> {
    __device__
    float operator()(unsigned int thread_id) {
        float sum = 0;
        unsigned int N = 10000; // samples per thread

        unsigned int seed = thread_id;

        curandState s;

        // seed a random number generator
        curand_init(seed, 0, 0, &s);

        // take N samples in a quarter circle
        for (unsigned int i = 0; i < N; ++i) {
            // draw a sample from the unit square
            float x = curand_uniform(&s);
            float y = curand_uniform(&s);

            // measure distance from the origin
            float dist = sqrtf(x * x + y * y);

            // add 1.0f if (u0,u1) is inside the quarter circle
            if (dist <= 1.0f)
                sum += 1.0f;
        }

        // multiply by 4 to get the area of the whole circle
        sum *= 4.0f;

        // divide by N
        return sum / N;
    }
};

int main(void) {
    // use 30K independent seeds
    int M = 30000;

    //total operations N from each call to estimate_pi and with M calls = M * N
    float estimate = thrust::transform_reduce(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(M),
                                              estimate_pi(), 0.0f, thrust::plus<float>());

    estimate /= M;

    std::cout << std::setprecision(4);
    std::cout << "pi is approximately ";
    std::cout << estimate << std::endl;
    return 0;
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