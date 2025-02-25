#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>

#define TILE_SIZE 256

std::vector<float> load_audio(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file");
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg() / sizeof(float);
    file.seekg(0, std::ios::beg);

    std::vector<float> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));
    return data;
}

__global__ void conv1d(float* input, float* kernel, float* output, int input_size, int kernel_size) {
    extern __shared__ float shared_mem[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int shared_offset = kernel_size / 2;

    if (tid < input_size) {
        shared_mem[threadIdx.x + shared_offset] = input[tid];
    }

    if (threadIdx.x < shared_offset) {
        if (tid >= shared_offset)
            shared_mem[threadIdx.x] = input[tid - shared_offset];
        else
            shared_mem[threadIdx.x] = 0.0f;
    }
    if (threadIdx.x + shared_offset + blockDim.x < input_size) {
        shared_mem[threadIdx.x + shared_offset + blockDim.x] = input[tid + blockDim.x];
    }

    __syncthreads();

    if (tid < input_size - kernel_size + 1) {
        float sum = 0.0f;
        for (int i = 0; i < kernel_size; i++) {
            sum += shared_mem[threadIdx.x + i] * kernel[i];
        }
        output[tid] = sum;
    }
}

void run_convolution(const float* h_input, const float* h_kernel, float* h_output, int input_size, int kernel_size) {
    float *d_input, *d_kernel, *d_output;
    size_t input_bytes = input_size * sizeof(float);
    size_t kernel_bytes = kernel_size * sizeof(float);
    size_t output_bytes = (input_size - kernel_size + 1) * sizeof(float);

    cudaMalloc((void**)&d_input, input_bytes);
    cudaMalloc((void**)&d_kernel, kernel_bytes);
    cudaMalloc((void**)&d_output, output_bytes);

    cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (input_size + threads - 1) / threads;
    size_t shared_mem_size = (threads + kernel_size) * sizeof(float);

    conv1d<<<blocks, threads, shared_mem_size>>>(d_input, d_kernel, d_output, input_size, kernel_size);

    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

int main() {
    std::vector<float> audio_data = load_audio("audio_input.bin");
    int input_size = audio_data.size();
    
    std::vector<float> kernel = {0.2f, 0.5f, 0.3f};  // Emphasizes low frequencies
    // std::vector<float> kernel = {-1.0f, 2.0f, -1.0f};  // Boosts high frequencies
    // std::vector<float> kernel = {-0.5f, 1.0f, -0.5f};  // Sharpens transients
    // std::vector<float> kernel = {1.0f, 0.0f, 0.5f};  // Adds a small echo
    // std::vector<float> kernel = {1.0f, 0.8f, 0.6f, 0.4f, 0.2f};  // Simulates large room
    // std::vector<float> kernel = {0.5f, 1.0f, 0.5f, 0.0f, -0.5f, -1.0f, -0.5f}; //Sinusoidal Volume Modulation
    // std::vector<float> kernel = {2.0f, -1.0f, 2.0f};  // Clipping-like effect

    int kernel_size = kernel.size();

    std::vector<float> output(input_size - kernel_size + 1);
    
    run_convolution(audio_data.data(), kernel.data(), output.data(), input_size, kernel_size);

    std::cout << "Processing completed!" << std::endl;
}
