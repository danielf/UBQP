#ifndef BRUTE_H
#define BRUTE_H
void cuda_initialize(int n);

void cuda_finalize();

void cuda_brute_force(float* Q, float* b, int* ans, int n);
#endif
