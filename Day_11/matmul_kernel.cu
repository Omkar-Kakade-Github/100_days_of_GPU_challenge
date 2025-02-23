#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 32

__global__ void matmul_cuda(float *A, float *B, float *C, int N) {
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;
    
    float value = 0.0f;
    
    for (int i = 0; i < N / BLOCK_SIZE; ++i) {
        sA[ty][tx] = A[row * N + (i * BLOCK_SIZE + tx)];
        sB[ty][tx] = B[(i * BLOCK_SIZE + ty) * N + col];
        
        __syncthreads();
        
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            value += sA[ty][j] * sB[j][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

extern "C" void run_cuda(float *h_A, float *h_B, float *h_C, int N, float *elapsed_time) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    matmul_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed_time, start, stop);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
