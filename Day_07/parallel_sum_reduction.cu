#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define BLOCK_SIZE 1024
#define CHUNK_SIZE (1 << 27)  // 2^27 elements (fits in ~2GB)

__global__ void reduceSum(const double *input, double *partialSums, int N) {
    __shared__ double sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? input[i] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partialSums[blockIdx.x] = sdata[0];
    }
}

double cpuSum(const double *array, int N) {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += array[i];
    }
    return sum;
}

int main() {
    int sizes[] = {1 << 10, 1 << 15, 1 << 20, 1 << 25, 1 << 30};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];

        if (N > CHUNK_SIZE) {
            std::cout << "N: " << N << " - Processing in chunks (Memory limitation workaround)\n";
        }

        double *h_input = new double[N];
        for (int i = 0; i < N; i++) h_input[i] = 1.0;

        // ---- CPU Benchmark ----
        auto start_cpu = std::chrono::high_resolution_clock::now();
        double cpu_result = cpuSum(h_input, N);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

        // ---- GPU Processing ----
        double gpu_result = 0.0;
        double total_gpu_time = 0.0;
        int processed = 0;

        while (processed < N) {
            int chunkSize = std::min(CHUNK_SIZE, N - processed);
            int numBlocks = (chunkSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

            double *d_input, *d_partialSums;
            cudaMalloc(&d_input, chunkSize * sizeof(double));
            cudaMalloc(&d_partialSums, numBlocks * sizeof(double));

            cudaMemcpy(d_input, h_input + processed, chunkSize * sizeof(double), cudaMemcpyHostToDevice);

            auto start_gpu = std::chrono::high_resolution_clock::now();
            reduceSum<<<numBlocks, BLOCK_SIZE>>>(d_input, d_partialSums, chunkSize);
            cudaDeviceSynchronize();
            auto end_gpu = std::chrono::high_resolution_clock::now();

            double gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();
            total_gpu_time += gpu_time;

            double *h_partialSums = new double[numBlocks];
            cudaMemcpy(h_partialSums, d_partialSums, numBlocks * sizeof(double), cudaMemcpyDeviceToHost);

            gpu_result += cpuSum(h_partialSums, numBlocks);

            delete[] h_partialSums;
            cudaFree(d_input);
            cudaFree(d_partialSums);

            processed += chunkSize;
        }

        // Print results
        std::cout << "N: " << N << " | CPU Sum: " << cpu_result << " (" << cpu_time << " ms)";
        std::cout << " | GPU Sum: " << gpu_result << " (" << total_gpu_time << " ms)" << std::endl;

        // Free Memory
        delete[] h_input;
    }

    return 0;
}
