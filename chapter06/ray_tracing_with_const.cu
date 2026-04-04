#include "../common/book.h"
#include "../common/cpu_bitmap.h"
#include <cmath>
#include <ctime>

#define INF 2e10f
#define SPHERES 20
#define DIM 1024
#define rnd(x) (x * rand() / RAND_MAX)

struct Sphere {
    float r, g, b;
    float radius;
    float x, y, z;

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

__constant__ Sphere dev_spheres[SPHERES];

__global__ void kernel(unsigned char *ptr) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = x + y * gridDim.x * blockDim.x;

    float half = DIM / 2.0f;
    float ox = (x - half);
    float oy = (y - half);

    float r = 0, g = 0, b = 0;
    float minz = INF;
    for (int i = 0; i < SPHERES; ++i) {
        float n;
        float t = dev_spheres[i].hit(ox, oy, &n);
        if (t < minz) {
            minz = t;
            float fscale = n;
            r = dev_spheres[i].r * fscale;
            g = dev_spheres[i].g * fscale;
            b = dev_spheres[i].b * fscale;
        }
    }

    ptr[idx * 4 + 0] = int(r * 255);
    ptr[idx * 4 + 1] = int(g * 255);
    ptr[idx * 4 + 2] = int(b * 255);
    ptr[idx * 4 + 3] = 255;
}

int main(void) {
    srand((unsigned)time(nullptr));

    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;
    HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));

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

    HANDLE_ERROR(cudaMemcpyToSymbol(dev_spheres, temp_s, SPHERES * sizeof(Sphere)));
    free(temp_s);

    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    dim3 threads(16, 16);
    dim3 blocks(DIM / 16, DIM / 16);
    kernel<<<blocks, threads>>>(dev_bitmap);

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float times;
    HANDLE_ERROR(cudaEventElapsedTime(&times, start, stop));
    printf("Time to generate: %.3f\n", times);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

    bitmap.display_and_exit();

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    cudaFree(dev_bitmap);

    return 0;
}