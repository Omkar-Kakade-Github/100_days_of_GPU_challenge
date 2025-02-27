#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 256
#define SOFTENING 1e-9f

struct Body {
    float x, y, z, vx, vy, vz, mass;
};

__device__ void bodyInteraction(Body &a, const Body &b) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;
    float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
    float invDist = rsqrtf(distSqr);
    float invDistCube = invDist * invDist * invDist;
    
    float force = b.mass * invDistCube;
    a.vx += dx * force;
    a.vy += dy * force;
    a.vz += dz * force;
}

__global__ void computeForces(Body *bodies, int numBodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;
    
    Body myBody = bodies[i];
    for (int j = 0; j < numBodies; j++) {
        if (i != j) {
            bodyInteraction(myBody, bodies[j]);
        }
    }
    bodies[i] = myBody;
}

__global__ void updatePositions(Body *bodies, float dt, int numBodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;
    
    bodies[i].x += bodies[i].vx * dt;
    bodies[i].y += bodies[i].vy * dt;
    bodies[i].z += bodies[i].vz * dt;
}

void simulate(Body *h_bodies, int numBodies, float dt, int numIterations) {
    Body *d_bodies;
    cudaMalloc(&d_bodies, numBodies * sizeof(Body));
    cudaMemcpy(d_bodies, h_bodies, numBodies * sizeof(Body), cudaMemcpyHostToDevice);
    
    int gridSize = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int iter = 0; iter < numIterations; iter++) {
        computeForces<<<gridSize, BLOCK_SIZE>>>(d_bodies, numBodies);
        updatePositions<<<gridSize, BLOCK_SIZE>>>(d_bodies, dt, numBodies);
        
        if (iter % 10 == 0) {
            cudaMemcpy(h_bodies, d_bodies, numBodies * sizeof(Body), cudaMemcpyDeviceToHost);
            std::cout << "Iteration " << iter << " completed. Sample body position: "
                      << h_bodies[0].x << ", " << h_bodies[0].y << ", " << h_bodies[0].z << std::endl;
        }
    }
    
    cudaMemcpy(h_bodies, d_bodies, numBodies * sizeof(Body), cudaMemcpyDeviceToHost);
    cudaFree(d_bodies);
}

int main() {
    const int numBodies = 4096;
    const float dt = 0.01f;
    const int numIterations = 1000;
    
    Body *h_bodies = new Body[numBodies];
    for (int i = 0; i < numBodies; i++) {
        h_bodies[i].x = rand() / (float)RAND_MAX;
        h_bodies[i].y = rand() / (float)RAND_MAX;
        h_bodies[i].z = rand() / (float)RAND_MAX;
        h_bodies[i].vx = h_bodies[i].vy = h_bodies[i].vz = 0.0f;
        h_bodies[i].mass = 1.0f;
    }
    
    simulate(h_bodies, numBodies, dt, numIterations);
    
    delete[] h_bodies;
    return 0;
}
