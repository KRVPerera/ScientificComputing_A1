// Source: http://docs.nvidia.com/cuda/curand/index.html
// Changed the N to match 4096
// Changed the M to match 256*256
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
#include <A1Config.h>

float elapsed_time_msec(struct timespec *begin, struct timespec *end,
                        unsigned long *sec, unsigned long *nsec);


#define GET_TIME(x);    if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
                { perror("clock_gettime( ):"); exit(EXIT_FAILURE); }

#define PI 3.1415926535  // known value of pi

// we could vary M & N to find the perf sweet spot

#ifdef USE_DOUBLE
struct estimate_pi :
        public thrust::unary_function<unsigned int, double> {
    __device__
    float operator()(unsigned int thread_id) {
        double sum = 0;
        unsigned int N = TRIALS_PER_THREAD; // samples per thread , changed to 4096 from 10000

        unsigned int seed = thread_id;

        curandState s;

        // seed a random number generator
        curand_init(seed, 0, 0, &s);

        // take N samples in a quarter circle
        for (unsigned int i = 0; i < N; ++i) {
            // draw a sample from the unit square
            double x = curand_uniform(&s);
            double y = curand_uniform(&s);

            // measure distance from the origin
            double dist = sqrtf(x * x + y * y);

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
#else
struct estimate_pi :
        public thrust::unary_function<unsigned int, float> {
    __device__
    float operator()(unsigned int thread_id) {
        float sum = 0;
        unsigned int N = TRIALS_PER_THREAD; // samples per thread , changed to 4096 from 10000

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

#endif
int main(void) {
    // Variables to be used in time calculation
    struct timespec t0, t1;
    float comp_time;
    unsigned long sec, nsec;
    // use 30K independent seeds
    int M = BLOCKS*THREADS;  // changed to match 256*256 from 30000

#ifdef USE_DOUBLE
    std::cout << "Running DOUBLE Version" << std::endl;
#else
    std::cout << "Running FLOAT Version" << std::endl;
#endif
    printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n", TRIALS_PER_THREAD,
           BLOCKS, THREADS);
    GET_TIME(t0);
    //total operations N from each call to estimate_pi and with M calls, leaving us with total M * N calculations

#ifdef USE_DOUBLE
    double estimate = thrust::transform_reduce(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(M),
                                              estimate_pi(), 0.0f, thrust::plus<double>());

#else
    float estimate = thrust::transform_reduce(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(M),
                                              estimate_pi(), 0.0f, thrust::plus<float>());
#endif
    estimate /= M;

    GET_TIME(t1);
    comp_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);

    std::cout << "pi-thrust pi calculated in \t" << comp_time << "ms." << std::endl;
    std::cout << std::setprecision(7);
    std::cout << "pi-thrust  estimate of PI \t= " << estimate << " \t[error of " ;
    std::cout << std::setprecision(7) << std::fixed << estimate - PI << "]" << std::endl;
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