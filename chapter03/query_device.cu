#include "../common/book.h"
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int count = 0;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; ++i) {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        printf("--- General Information for device %d ---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("--- Memory Information for device %d ---\n", i);
        printf("Total global mem: %ld GB\n", prop.totalGlobalMem >> 30);
        printf("L2 cache size: %d KB\n", prop.l2CacheSize / 1024);
    }
    return 0;
}