#include "../common/book.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 32

__global__ void vector_add(int *a, int *b, int *c) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
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

    vector_add<<<1, N>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    free(a);
    free(b);
    free(c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}