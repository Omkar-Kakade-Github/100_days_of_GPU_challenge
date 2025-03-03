#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Scoring parameters
#define MATCH_SCORE 2
#define MISMATCH_PENALTY -1
#define GAP_PENALTY -1
#define BLOCK_SIZE 16

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA kernel for Smith-Waterman algorithm using diagonal wavefront pattern
__global__ void smithWatermanKernel(
    const char* query,         // Query sequence
    const char* reference,     // Reference sequence
    int queryLength,           // Length of query sequence
    int refLength,             // Length of reference sequence
    int* scoreMatrix,          // Output score matrix (flattened 2D array)
    int* maxScore,             // Pointer to store maximum score
    int* maxI,                 // Pointer to store i-index of max score
    int* maxJ)                 // Pointer to store j-index of max score
{
    // Shared memory for caching part of the computation
    __shared__ int sharedScores[BLOCK_SIZE + 1][BLOCK_SIZE + 1];
    
    // Initialize shared memory
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Each block processes a portion of the scoring matrix
    int startI = blockIdx.y * blockDim.y;
    int startJ = blockIdx.x * blockDim.x;
    
    // Process the matrix in diagonal wavefronts
    for (int diag = 0; diag < blockDim.x + blockDim.y - 1; diag++) {
        // Calculate the starting position for this thread in the current diagonal
        int i = ty;
        int j = tx;
        
        if (i + j == diag) {
            // Calculate actual position in the global matrix. The matrix uses 0-based indexing.
            int globalI = startI + i + 1;  // +1 for matrix offset
            int globalJ = startJ + j + 1;  // +1 for matrix offset
            
            // Check if we're within bounds
            if (globalI <= queryLength && globalJ <= refLength) {
                int score = 0;
                
                // Get the previous scores
                int diagScore = 0;
                int upScore = 0;
                int leftScore = 0;
                
                // Check if previous values are in shared memory or need to be loaded from global memory
                if (i > 0 && j > 0) {
                    diagScore = sharedScores[i-1][j-1];
                } else if (globalI > 1 && globalJ > 1) {
                    diagScore = scoreMatrix[(globalI-2) * (refLength+1) + (globalJ-2)];
                }
                
                if (i > 0) {
                    upScore = sharedScores[i-1][j];
                } else if (globalI > 1) {
                    upScore = scoreMatrix[(globalI-2) * (refLength+1) + (globalJ-1)];
                }
                
                if (j > 0) {
                    leftScore = sharedScores[i][j-1];
                } else if (globalJ > 1) {
                    leftScore = scoreMatrix[(globalI-1) * (refLength+1) + (globalJ-2)];
                }
                
                // Apply match/mismatch score
                int match = (query[globalI-1] == reference[globalJ-1]) ? MATCH_SCORE : MISMATCH_PENALTY;
                
                // Calculate the score for this cell
                diagScore += match;
                upScore += GAP_PENALTY;
                leftScore += GAP_PENALTY;
                
                // Find the maximum score
                score = max(0, max(diagScore, max(upScore, leftScore)));
                
                // Store in shared memory
                sharedScores[i][j] = score;
                
                // Store in global memory
                scoreMatrix[(globalI-1) * (refLength+1) + (globalJ-1)] = score;
                
                // Update maximum score (using atomic operations to avoid race conditions)
                if (score > 0) {
                    atomicMax(maxScore, score);
                    
                    // If this thread has the maximum score, update indices
                    if (score == *maxScore) {
                        atomicExch(maxI, globalI-1);
                        atomicExch(maxJ, globalJ-1);
                    }
                }
            }
        }
        
        // Synchronize threads before proceeding to next diagonal
        __syncthreads();
    }
}

// Host function to set up and launch the CUDA kernel
void runSmithWatermanGPU(const char* query, const char* reference) {
    int queryLength = strlen(query);
    int refLength = strlen(reference);
    
    // Allocate host memory
    int *h_scoreMatrix = (int*)malloc((queryLength+1) * (refLength+1) * sizeof(int));
    int h_maxScore = 0;
    int h_maxI = 0;
    int h_maxJ = 0;
    
    // Initialize host score matrix
    memset(h_scoreMatrix, 0, (queryLength+1) * (refLength+1) * sizeof(int));
    
    // Allocate device memory
    char *d_query, *d_reference;
    int *d_scoreMatrix, *d_maxScore, *d_maxI, *d_maxJ;
    
    CUDA_CHECK(cudaMalloc((void**)&d_query, queryLength * sizeof(char)));
    CUDA_CHECK(cudaMalloc((void**)&d_reference, refLength * sizeof(char)));
    CUDA_CHECK(cudaMalloc((void**)&d_scoreMatrix, (queryLength+1) * (refLength+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_maxScore, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_maxI, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_maxJ, sizeof(int)));
    
    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_query, query, queryLength * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reference, reference, refLength * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scoreMatrix, h_scoreMatrix, (queryLength+1) * (refLength+1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_maxScore, &h_maxScore, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_maxI, &h_maxI, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_maxJ, &h_maxJ, sizeof(int), cudaMemcpyHostToDevice));
    
    // Set up grid and block dimensions
    // Calculate grid size based on sequence lengths
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (refLength + blockDim.x - 1) / blockDim.x,
        (queryLength + blockDim.y - 1) / blockDim.y
    );
    
    // Launch kernel
    smithWatermanKernel<<<gridDim, blockDim>>>(
        d_query, d_reference, queryLength, refLength,
        d_scoreMatrix, d_maxScore, d_maxI, d_maxJ
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_scoreMatrix, d_scoreMatrix, (queryLength+1) * (refLength+1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_maxScore, d_maxScore, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_maxI, d_maxI, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_maxJ, d_maxJ, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Print results
    printf("Maximum alignment score: %d\n", h_maxScore);
    printf("At position: Query[%d], Reference[%d]\n", h_maxI, h_maxJ);
    
    // Print score matrix for debugging (only for small sequences)
    if (queryLength < 20 && refLength < 20) {
        printf("\nScore Matrix:\n");
        for (int i = 0; i < queryLength + 1; i++) {
            for (int j = 0; j < refLength + 1; j++) {
                printf("%3d ", h_scoreMatrix[i * (refLength+1) + j]);
            }
            printf("\n");
        }
    }
    
    // Free device memory
    cudaFree(d_query);
    cudaFree(d_reference);
    cudaFree(d_scoreMatrix);
    cudaFree(d_maxScore);
    cudaFree(d_maxI);
    cudaFree(d_maxJ);
    
    // Free host memory
    free(h_scoreMatrix);
}

// Example usage
int main() {
    // Example DNA sequences
    const char* query = "ACGTGCTATGCAGT";
    const char* reference = "ACGTACGTAGCTGATCG";
    
    printf("Query sequence: %s\n", query);
    printf("Reference sequence: %s\n", reference);
    
    // Run Smith-Waterman algorithm on GPU
    runSmithWatermanGPU(query, reference);
    
    return 0;
}
