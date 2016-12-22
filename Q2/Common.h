//
// Created by krv on 12/22/16.
//

#ifndef CS4552_A1_COMMON_H
#define CS4552_A1_COMMON_H


#define GET_TIME(x); x

float elapsed_time_msec(struct timespec *begin, struct timespec *end,
                        unsigned long *sec, unsigned long *nsec);


#endif //CS4552_A1_COMMON_H
