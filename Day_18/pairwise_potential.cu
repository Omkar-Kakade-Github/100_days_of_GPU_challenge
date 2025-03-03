#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// -----------------------------------------------------------------------------
// Kernel: Computes the pairwise potential for each particle.
// For each particle i, loop over all other particles j and compute:
//
//   sum_i = Σ_{j≠i} (m_i * m_j) / r(i,j)
// 
// where r(i,j) is the distance between particles i and j.
// Each thread writes its partial sum into partial_sums[i].
// -----------------------------------------------------------------------------
__global__ void pairwise_potential(const float* positions,
                                   const float* masses,
                                   float* partial_sums,
                                   int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Load the position and mass for particle i
    float xi = positions[3 * i + 0];
    float yi = positions[3 * i + 1];
    float zi = positions[3 * i + 2];
    float mi = masses[i];

    float sum_i = 0.0f;

    // Loop over all particles j to compute pairwise interactions
    for (int j = 0; j < N; j++) {
        if (j != i) {
            float xj = positions[3 * j + 0];
            float yj = positions[3 * j + 1];
            float zj = positions[3 * j + 2];
            float mj = masses[j];

            // Compute distance between particle i and j
            float dx = xj - xi;
            float dy = yj - yi;
            float dz = zj - zi;
            float r = sqrtf(dx * dx + dy * dy + dz * dz);

            // Avoid division by zero (or very small distances)
            if (r > 1e-5f) {
                sum_i += (mi * mj) / r;
            }
        }
    }
    // Write the partial sum for particle i
    partial_sums[i] = sum_i;
}

// -----------------------------------------------------------------------------
// Main function: sets up data, launches the kernel, reduces the results,
// and prints the total potential energy.
// -----------------------------------------------------------------------------
int main()
{
    // Number of particles (small number for demonstration)
    const int N = 16;
    const int numPositions = 3 * N; // each particle has x, y, z components

    size_t positionsSize = numPositions * sizeof(float);
    size_t massesSize    = N * sizeof(float);
    size_t partialSize   = N * sizeof(float);

    // Allocate host memory
    float* h_positions = (float*)malloc(positionsSize);
    float* h_masses    = (float*)malloc(massesSize);
    float* h_partial   = (float*)malloc(partialSize);

    // Initialize host arrays
    // For positions, assign random values in the range [0, 1)
    // For masses, assign 1.0 for all particles
    for (int i = 0; i < N; i++) {
        h_positions[3 * i + 0] = (float)rand() / RAND_MAX; // x
        h_positions[3 * i + 1] = (float)rand() / RAND_MAX; // y
        h_positions[3 * i + 2] = (float)rand() / RAND_MAX; // z
        h_masses[i] = 1.0f;
    }

    // Allocate device memory
    float* d_positions = nullptr;
    float* d_masses    = nullptr;
    float* d_partial   = nullptr;
    cudaMalloc((void**)&d_positions, positionsSize);
    cudaMalloc((void**)&d_masses, massesSize);
    cudaMalloc((void**)&d_partial, partialSize);

    // Copy host data to device
    cudaMemcpy(d_positions, h_positions, positionsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_masses, h_masses, massesSize, cudaMemcpyHostToDevice);

    // Configure and launch the kernel
    int threadsPerBlock = 128;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    pairwise_potential<<<blocks, threadsPerBlock>>>(d_positions, d_masses, d_partial, N);
    cudaDeviceSynchronize();

    // Copy partial sums back to host
    cudaMemcpy(h_partial, d_partial, partialSize, cudaMemcpyDeviceToHost);

    // Reduce the partial sums on the host.
    // Each pair interaction is counted twice (i->j and j->i), so multiply by 0.5.
    float totalPotential = 0.0f;
    for (int i = 0; i < N; i++) {
        totalPotential += h_partial[i];
    }
    totalPotential *= 0.5f;

    // Print the result
    printf("Total potential energy: %f\n", totalPotential);

    // Free device and host memory
    cudaFree(d_positions);
    cudaFree(d_masses);
    cudaFree(d_partial);
    free(h_positions);
    free(h_masses);
    free(h_partial);

    return 0;
}
