#include "sgemm_kernels.h"
#include <cstdio>

__global__ void matrix_multiplication_kernel(int M, int N, int K, 
                                   float alpha, float *A, float *B, 
                                   float beta, float *C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * sum + beta * C[x * N + y];
    }
}

