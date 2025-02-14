__global__ void sgemm_global_mem_coalesce(int M, int N, int K, 
                                         float alpha, float *A, float *B, 
                                         float beta, float *C) {
    const int cRow = blockIdx.x * 32 + (threadIdx.x / 32);
    const int cCol = blockIdx.y * 32 + (threadIdx.x % 32);

    if (cRow < M && cCol < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[cRow * K + i] * B[i * N + cCol];
        }

        C[cRow * N + cCol] = alpha * sum + beta * C[cRow * N + cCol];
    }
}
