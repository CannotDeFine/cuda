#include "../common/book.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define DIM 1000

// Minimal complex number helper for Julia set iteration on the GPU.
struct cuComplex {
    float r;
    float i;

    __device__ cuComplex(float a, float b) : r(a), i(b) {}

    __device__ float magnitude(void) {
        return r * r + i * i;
    }

    __device__ cuComplex operator*(const cuComplex &a) {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    __device__ cuComplex operator+(const cuComplex &a) {
        return cuComplex(r + a.r, i + a.i);
    }
};

// Maps a pixel coordinate to the complex plane and tests Julia set membership.
__device__ int julia(int x, int y) {

    const float scale = 1.5;
    const float half = DIM / 2.0f;

    float jx = scale * (half - x) / half;
    float jy = scale * (half - y) / half;

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    for (int i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude() > 1000) {
            return 0;
        }
    }

    return 1;
}

// Each block computes one pixel and writes it as RGBA.
__global__ void kernel(unsigned char *ptr) {

    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = y + x * gridDim.y;

    int juliaValue = julia(x, y);
    ptr[offset * 4 + 0] = 255 * juliaValue;
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

static void write_ppm(const char *path, const unsigned char *pixels, int width, int height) {
    FILE *fp = fopen(path, "wb");
    if (fp == NULL) {
        fprintf(stderr, "failed to open %s for writing\n", path);
        exit(EXIT_FAILURE);
    }

    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    for (int i = 0; i < width * height; ++i) {
        fwrite(&pixels[i * 4], 1, 3, fp);
    }
    fclose(fp);
}

int main() {
    const size_t image_size = DIM * DIM * 4;
    unsigned char *pixels = (unsigned char *)malloc(image_size);
    HANDLE_NULL(pixels);

    unsigned char *dev_ptr = nullptr;
    HANDLE_ERROR(cudaMalloc((void **)&dev_ptr, image_size));

    // Launch one block per pixel so blockIdx maps directly to image coordinates.
    dim3 grid(DIM, DIM);
    kernel<<<grid, 1>>>(dev_ptr);

    // Validate the launch before copying the rendered image back to host memory.
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaMemcpy(pixels, dev_ptr, image_size, cudaMemcpyDeviceToHost));

    write_ppm("chapter04/julia.ppm", pixels, DIM, DIM);
    HANDLE_ERROR(cudaFree(dev_ptr));
    free(pixels);

    printf("wrote chapter04/julia.ppm\n");

    return 0;
}
