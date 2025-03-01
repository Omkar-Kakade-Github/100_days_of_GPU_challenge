#include <stdio.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// 1. Define stencil coefficients in constant memory:
// -----------------------------------------------------------------------------
__constant__ float c0 = 0.5f;
__constant__ float c1 = 0.1f;
__constant__ float c2 = 0.1f;
__constant__ float c3 = 0.1f;
__constant__ float c4 = 0.1f;
__constant__ float c5 = 0.05f;
__constant__ float c6 = 0.05f;

// -----------------------------------------------------------------------------
// 2. Define the stencil kernel
// -----------------------------------------------------------------------------
__global__ void stencil_kernel(const float* __restrict__ in,
                                     float* __restrict__ out,
                               unsigned int N)
{
    // Compute 3D indices (i, j, k) from the block/thread indices
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    // Only compute if we have valid neighbors in all directions
    // (i.e., not on the boundary).
    if (i >= 1 && i < N - 1 &&
        j >= 1 && j < N - 1 &&
        k >= 1 && k < N - 1)
    {
        // Convert 3D (i, j, k) to 1D index: i*N*N + j*N + k
        unsigned int idx    = i*N*N + j*N + k;
        unsigned int idx_px = i*N*N + j*N + (k + 1);
        unsigned int idx_mx = i*N*N + j*N + (k - 1);
        unsigned int idx_py = i*N*N + (j + 1)*N + k;
        unsigned int idx_my = i*N*N + (j - 1)*N + k;
        unsigned int idx_pz = (i + 1)*N*N + j*N + k;
        unsigned int idx_mz = (i - 1)*N*N + j*N + k;

        // 7-point stencil update
        out[idx] = c0 * in[idx]
                 + c1 * in[idx_mx]
                 + c2 * in[idx_px]
                 + c3 * in[idx_my]
                 + c4 * in[idx_py]
                 + c5 * in[idx_mz]
                 + c6 * in[idx_pz];
    }
}

// -----------------------------------------------------------------------------
// 3. Main function
// -----------------------------------------------------------------------------
int main()
{
    // Choose a size for the 3D data (N x N x N)
    const unsigned int N = 16;  // for demonstration; can be larger
    const size_t size = N * N * N * sizeof(float);

    // -------------------------------------------------------------------------
    // 3.1. Allocate host memory and initialize input data
    // -------------------------------------------------------------------------
    float* h_in  = (float*)malloc(size);
    float* h_out = (float*)malloc(size);

    // Fill h_in with some values
    for (unsigned int i = 0; i < N * N * N; ++i) {
        h_in[i] = (float)i / (N*N*N);  // range [0..1)
    }

    // Initialize h_out to zero
    for (unsigned int i = 0; i < N * N * N; ++i) {
        h_out[i] = 0.0f;
    }

    // -------------------------------------------------------------------------
    // 3.2. Allocate device memory
    // -------------------------------------------------------------------------
    float *d_in  = nullptr;
    float *d_out = nullptr;
    cudaMalloc((void**)&d_in,  size);
    cudaMalloc((void**)&d_out, size);

    // -------------------------------------------------------------------------
    // 3.3. Copy data from host to device
    // -------------------------------------------------------------------------
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // -------------------------------------------------------------------------
    // 3.4. Define block and grid dimensions
    // -------------------------------------------------------------------------
    // For a 3D problem of size N, we choose a block dimension (8x8x8) or similar.
    // Then compute the grid dimension to cover N in each dimension.
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y,
                 (N + blockDim.z - 1) / blockDim.z);

    // -------------------------------------------------------------------------
    // 3.5. Launch the kernel
    // -------------------------------------------------------------------------
    stencil_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Cleanup and return
        cudaFree(d_in);
        cudaFree(d_out);
        free(h_in);
        free(h_out);
        return 1;
    }

    // -------------------------------------------------------------------------
    // 3.6. Copy the results back to the host
    // -------------------------------------------------------------------------
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // -------------------------------------------------------------------------
    // 3.7. Use the output (example: compute sum of all output elements)
    // -------------------------------------------------------------------------
    double sum = 0.0;
    for (unsigned int i = 0; i < N * N * N; ++i) {
        sum += (double)h_out[i];
    }
    printf("Sum of output array = %f\n", sum);

    // -------------------------------------------------------------------------
    // 3.8. Cleanup
    // -------------------------------------------------------------------------
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
