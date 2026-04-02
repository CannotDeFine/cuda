#include <ctime>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define INF 2e10f
#define SPHERES 20
#define DIM 1024
#define rnd(x) (x * rand() / RAND_MAX)

// A sphere with color and position in a very small scene description.
struct Sphere {
    float r, b, g;
    float radius;
    float x, y, z;

    // Tests whether the ray through screen position (ox, oy) hits this sphere.
    // Returns the z-depth of the hit point and stores a simple lighting term in n.
    __device__ float hit(float ox, float oy, float *n) const {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return z - dz;
        }
        return INF;
    }
};

// Each thread shades one output pixel by finding the front-most sphere on that ray.
__global__ void kernel(unsigned char *ptr, const Sphere *s) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int offset = x + y * gridDim.x * blockDim.x;

    // Center the image plane so the scene is rendered around the origin.
    float ox = (x - DIM / 2);
    float oy = (y - DIM / 2);

    float r = 0, g = 0, b = 0;
    float minz = INF;
    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = s[i].hit(ox, oy, &n);
        if (t < minz) {
            minz = t;
            // Use the normalized z component as a simple diffuse lighting factor.
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
        }
    }

    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}

int main() {
    srand((unsigned)time(nullptr));

    cudaEvent_t start, stop;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;
    Sphere *dev_spheres;
    HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void **)&dev_spheres, sizeof(Sphere) * SPHERES));

    // Build a random scene on the host, then copy it to device memory.
    Sphere *temp_s = (Sphere *)malloc(sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 500;
        temp_s[i].radius = rnd(100.0f) + 20;
    }

    HANDLE_ERROR(cudaMemcpy(dev_spheres, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice));
    free(temp_s);

    // Launch a 2D grid so each thread computes one pixel of the final image.
    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel<<<grids, threads>>>(dev_bitmap, dev_spheres);

    // Wait for rendering to finish before copying the image back for display.
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(
        cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

    bitmap.display_and_exit();

    cudaFree(dev_bitmap);
    cudaFree(dev_spheres);

    return 0;
}
