#include "../common/book.h"

#include <cuda_runtime.h>
#include <cstdio>

#define N (33 * 1024)

__global__ void kernel(int *a, int *b, int *c) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < N) {
        c[idx] = a[idx] + b[idx];
        idx += blockDim.x * gridDim.x;
    }
}

int main() {
    int *a = (int *)malloc(N * sizeof(int));
    int *b = (int *)malloc(N * sizeof(int));
    int *c = (int *)malloc(N * sizeof(int));

    int *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
    HANDLE_ERROR(cudaMalloc((void **)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_c, N * sizeof(int)));

    for (int i = 0; i < N; i++) {
        a[i] = 1;
        b[i] = 2;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    kernel<<<128, 128>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
    
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (a[i] + b[i] != c[i]) {
            printf("Error: %d + %d = %d\n", a[i], b[i], c[i]);
            success = false;
            break;
        }
    }
    if (success) printf("We did it!\n");

    free(a);
    free(b);
    free(c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}