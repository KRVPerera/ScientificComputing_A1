//
// Created by krv on 12/22/16.
//
#include <time.h>
#include <errno.h>
#ifndef CS4552_A1_COMMON_H
#define CS4552_A1_COMMON_H


#define GET_TIME(x);	if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
				{ perror("clock_gettime( ):"); exit(EXIT_FAILURE); }

float elapsed_time_msec(struct timespec *begin, struct timespec *end,
                        unsigned long *sec, unsigned long *nsec);


#endif //CS4552_A1_COMMON_H
