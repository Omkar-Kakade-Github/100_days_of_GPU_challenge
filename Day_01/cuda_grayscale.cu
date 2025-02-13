// CUDA kernel for RGB to grayscale conversion
__global__ void rgbToGrayscale(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Calculate the index for the current pixel
        int grayOffset = y * width + x;

        // Get the RGB values
        unsigned char r = input[grayOffset * 3];
        unsigned char g = input[grayOffset * 3 + 1];
        unsigned char b = input[grayOffset * 3 + 2];

        // Convert to grayscale
        output[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}
