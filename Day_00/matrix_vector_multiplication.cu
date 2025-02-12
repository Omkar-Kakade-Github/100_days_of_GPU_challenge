// Given a matrix A of shape (M,K) and an input vector B of shape (K,1)

__global__ void sgemv_kernel(int M, int K, float alpha, float *A, float *B, float beta, float *C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < M) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[x * K + i] * B[i];
        }
        C[x] = alpha * sum + beta * C[x];
    }
}
