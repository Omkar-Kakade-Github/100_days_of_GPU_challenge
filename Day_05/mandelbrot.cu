#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 4096
#define HEIGHT 4096
#define MAX_ITER 5000

__global__ void mandelbrotKernel(unsigned char *image, int width, int height, double xMin, double xMax, double yMin, double yMax) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    double real = xMin + (x / (double)width) * (xMax - xMin);
    double imag = yMin + (y / (double)height) * (yMax - yMin);
    
    double z_real = real, z_imag = imag;
    int iter;
    for (iter = 0; iter < MAX_ITER; iter++) {
        double r2 = z_real * z_real, i2 = z_imag * z_imag;
        if (r2 + i2 > 4.0) break;
        z_imag = 2.0 * z_real * z_imag + imag;
        z_real = r2 - i2 + real;
    }
    
    int index = (y * width + x);
    image[index] = (unsigned char)(255 * iter / MAX_ITER);
}

void generateMandelbrot(unsigned char *h_image) {
    unsigned char *d_image;
    size_t size = WIDTH * HEIGHT * sizeof(unsigned char);
    cudaMalloc(&d_image, size);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);
    
    mandelbrotKernel<<<gridSize, blockSize>>>(d_image, WIDTH, HEIGHT, -2.0f, 1.0f, -1.5f, 1.5f);
    cudaMemcpy(h_image, d_image, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_image);
}

int main() {
    unsigned char *h_image = (unsigned char*)malloc(WIDTH * HEIGHT * sizeof(unsigned char));
    generateMandelbrot(h_image);
    
    FILE *fp = fopen("mandelbrot.pgm", "wb");
    fprintf(fp, "P5\n%d %d\n255\n", WIDTH, HEIGHT);
    fwrite(h_image, sizeof(unsigned char), WIDTH * HEIGHT, fp);
    fclose(fp);
    
    free(h_image);
    printf("Mandelbrot set image saved as 'mandelbrot.pgm'\n");
    return 0;
}
