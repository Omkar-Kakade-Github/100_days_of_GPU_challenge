#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <time.h>

// Simulation constants
#define N 1024             // Number of bodies
#define DT 0.01f           // Time step
#define G 6.67430e-11f      // Gravitational constant
#define STEPS 1000         // Number of simulation steps
#define SOFTENING 1e-9f    // Softening factor to avoid singularity

#define CUDA_CHECK(call)                                                          \
    {                                                                             \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    }

struct Body {
    float x, y, z;      // Position
    float vx, vy, vz;   // Velocity
    float mass;         // Mass
};

__global__ void updateBodies(Body* bodies, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    for (int j = 0; j < n; j++) {
        if (i == j) continue; 

        float dx = bodies[j].x - bodies[i].x;
        float dy = bodies[j].y - bodies[i].y;
        float dz = bodies[j].z - bodies[i].z;

        float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;

        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        float force = G * bodies[i].mass * bodies[j].mass * invDist3;

        fx += force * dx;
        fy += force * dy;
        fz += force * dz;
    }

    // Update velocity (acceleration = force/mass)
    bodies[i].vx += dt * fx / bodies[i].mass;
    bodies[i].vy += dt * fy / bodies[i].mass;
    bodies[i].vz += dt * fz / bodies[i].mass;

    // Update position using the new velocity
    bodies[i].x += dt * bodies[i].vx;
    bodies[i].y += dt * bodies[i].vy;
    bodies[i].z += dt * bodies[i].vz;
}

int main() {
    Body* h_bodies = (Body*)malloc(N * sizeof(Body));
    if (h_bodies == nullptr) {
        fprintf(stderr, "Error allocating host memory.\n");
        return EXIT_FAILURE;
    }

    srand(time(NULL));

    // Initialize bodies with random positions, velocities, and masses
    for (int i = 0; i < N; i++) {
        h_bodies[i].x = (float)rand() / RAND_MAX * 100.0f - 50.0f;
        h_bodies[i].y = (float)rand() / RAND_MAX * 100.0f - 50.0f;
        h_bodies[i].z = (float)rand() / RAND_MAX * 100.0f - 50.0f;
        h_bodies[i].vx = (float)rand() / RAND_MAX * 1.0f - 0.5f;
        h_bodies[i].vy = (float)rand() / RAND_MAX * 1.0f - 0.5f;
        h_bodies[i].vz = (float)rand() / RAND_MAX * 1.0f - 0.5f;
        h_bodies[i].mass = (float)rand() / RAND_MAX * 1e21f + 1e20f;  // mass between 1e20 and 1.1e21
    }

    Body* d_bodies;
    CUDA_CHECK(cudaMalloc((void**)&d_bodies, N * sizeof(Body)));

    CUDA_CHECK(cudaMemcpy(d_bodies, h_bodies, N * sizeof(Body), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    for (int step = 0; step < STEPS; step++) {
        updateBodies<<<gridSize, blockSize>>>(d_bodies, N, DT);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(h_bodies, d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 10; i++) {
        printf("Body %d: Position (%.5f, %.5f, %.5f)  Velocity (%.5f, %.5f, %.5f)\n",
               i, h_bodies[i].x, h_bodies[i].y, h_bodies[i].z,
               h_bodies[i].vx, h_bodies[i].vy, h_bodies[i].vz);
    }

    CUDA_CHECK(cudaFree(d_bodies));
    free(h_bodies);

    return 0;
}
