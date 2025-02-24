#include <cuda_runtime.h>
#include <cstdio>
#include "stb_image.h"
#include "stb_image_write.h"

__global__ void convolution_2D(float *N, float *F, float *P, int r, int width, int height) {

    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    if (outCol >= width || outRow >= height) return; // Avoid out-of-bounds threads

    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2*r+1; fRow++) {
        for (int fCol = 0; fCol < 2*r+1; fCol++) {
            int inRow = outRow + fRow - r;
            int inCol = outCol + fCol - r;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += N[inRow * width + inCol] * F[fRow * r + fCol];
            }
        }
    }
    P[outRow * width + outCol] = Pvalue;
    
}

void applyConvolution(const char *inputImage, const char *outputImage) {
    int width, height, channels;
    
    // Load image using stb_image.h
    unsigned char *image = stbi_load(inputImage, &width, &height, &channels, 0);
    if (!image) {
        printf("Failed to load image!\n");
        return;
    }

    printf("Image Loaded: %dx%d, Channels: %d\n", width, height, channels);

    // Convert image to grayscale
    float *grayImage = (float *)malloc(width * height * sizeof(float));
    for (int i = 0; i < width * height; i++) {
        int r = image[i * channels];
        int g = image[i * channels + 1];
        int b = image[i * channels + 2];
        grayImage[i] = 0.299f * r + 0.587f * g + 0.114f * b; // Grayscale conversion
    }

    // Define filter
    int r = 3;
    // float box_blur[] = {
    //     1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
    //     1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
    //     1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
    //     1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
    //     1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
    //     1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
    //     1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f

    // };

    // float gaussian_blur[] = {
    //     1/140.0f, 1/70.0f, 1/56.0f, 1/52.0f, 1/56.0f, 1/70.0f, 1/140.0f,
    //     1/70.0f, 1/35.0f, 4/56.0f, 4/52.0f, 4/56.0f, 1/35.0f, 1/70.0f,
    //     1/56.0f, 4/56.0f, 6/56.0f, 6/52.0f, 6/56.0f, 4/56.0f, 1/56.0f,
    //     1/52.0f, 4/52.0f, 6/52.0f, 8/52.0f, 6/52.0f, 4/52.0f, 1/52.0f,
    //     1/56.0f, 4/56.0f, 6/56.0f, 6/52.0f, 6/56.0f, 4/56.0f, 1/56.0f,
    //     1/70.0f, 1/35.0f, 4/56.0f, 4/52.0f, 4/56.0f, 1/35.0f, 1/70.0f,
    //     1/140.0f, 1/70.0f, 1/56.0f, 1/52.0f, 1/56.0f, 1/70.0f, 1/140.0f
    // };

    // float laplacian[] = {
    //       0,  0, -1, -1, -1,  0,  0,
    //       0, -1, -3, -3, -3, -1,  0,
    //      -1, -3,  0,  7,  0, -3, -1,
    //      -1, -3,  7, 24,  7, -3, -1,
    //      -1, -3,  0,  7,  0, -3, -1,
    //       0, -1, -3, -3, -3, -1,  0,
    //       0,  0, -1, -1, -1,  0,  0
    // };

    // float sharpen[] = {
    //       0,  0, -1, -1, -1,  0,  0,
    //       0, -1, -3, -3, -3, -1,  0,
    //      -1, -3,  0,  7,  0, -3, -1,
    //      -1, -3,  7, 36,  7, -3, -1,
    //      -1, -3,  0,  7,  0, -3, -1,
    //       0, -1, -3, -3, -3, -1,  0,
    //       0,  0, -1, -1, -1,  0,  0
    // };

    float emboss[] = {
         -4, -3, -2, -1,  0,  1,  2,
         -3, -2, -1,  0,  1,  2,  3,
         -2, -1,  0,  1,  2,  3,  4,
         -1,  0,  1,  2,  3,  4,  5,
          0,  1,  2,  3,  4,  5,  6,
          1,  2,  3,  4,  5,  6,  7,
          2,  3,  4,  5,  6,  7,  8
    };

    // Allocate device memory
    float *d_N, *d_F, *d_P;
    cudaMalloc(&d_N, width * height * sizeof(float));
    cudaMalloc(&d_F, (2 * r + 1) * (2 * r + 1) * sizeof(float));
    cudaMalloc(&d_P, width * height * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_N, grayImage, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, emboss, (2 * r + 1) * (2 * r + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Configure kernel launch
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    convolution_2D<<<gridSize, blockSize>>>(d_N, d_F, d_P, r, width, height);
    cudaDeviceSynchronize();

    // Copy result back
    float *h_P = (float *)malloc(width * height * sizeof(float));
    cudaMemcpy(h_P, d_P, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Normalize result to [0, 255]
    unsigned char *outputImageData = (unsigned char *)malloc(width * height);
    float minVal = h_P[0], maxVal = h_P[0];
    for (int i = 0; i < width * height; i++) {
        if (h_P[i] < minVal) minVal = h_P[i];
        if (h_P[i] > maxVal) maxVal = h_P[i];
    }
    for (int i = 0; i < width * height; i++) {
        outputImageData[i] = (unsigned char)(255 * (h_P[i] - minVal) / (maxVal - minVal));
    }

    // Save output using stb_image_write.h
    stbi_write_png(outputImage, width, height, 1, outputImageData, width);

    printf("Output saved to %s\n", outputImage);

    // Free memory
    stbi_image_free(image);
    free(grayImage);
    free(h_P);
    free(outputImageData);
    cudaFree(d_N);
    cudaFree(d_F);
    cudaFree(d_P);
}

int main() {
    applyConvolution("input.png", "output.png");
    return 0;
}
