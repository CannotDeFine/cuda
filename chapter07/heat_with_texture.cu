#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../common/book.h"
#include "../common/cpu_anim.h"

#define DIM 1024
#define PI 3.1415
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

struct DataBlock {
    unsigned char *output_bitmap;
    float *dev_in_src;
    float *dev_out_src;
    float *dev_const_src;
    CPUAnimBitmap *bitmap;
    cudaTextureObject_t const_tex;

    cudaEvent_t start, stop;
    float total_time;
    float frames;
};

__global__ void copy_const_kernel(float *iptr, cudaTextureObject_t const_tex) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = x + y * gridDim.x * blockDim.x;

    float value = tex1Dfetch<float>(const_tex, idx);
    if (value != 0.0f) iptr[idx] = value;
}

__global__ void blend_kernel(float *out_src, const float *in_src) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = x + y * gridDim.x * blockDim.x;

    int left = idx - 1;
    int right = idx + 1;
    if (x == 0) left++;
    if (x == DIM - 1) right--;

    int top = idx - DIM;
    int bottom = idx + DIM;
    if (y == 0) top += DIM;
    if (y == DIM - 1) bottom -= DIM;

    out_src[idx] = in_src[idx] +
        SPEED * (in_src[top] + in_src[bottom] + in_src[left] + in_src[right] -
                 in_src[idx] * 4);
}

void anim_gpu(DataBlock *d, int ticks) {
    (void)ticks;

    HANDLE_ERROR(cudaEventRecord(d->start, 0));
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    for (int i = 0; i < 90; i++) {
        copy_const_kernel<<<blocks, threads>>>(d->dev_in_src, d->const_tex);
        blend_kernel<<<blocks, threads>>>(d->dev_out_src, d->dev_in_src);
        swap(d->dev_in_src, d->dev_out_src);
    }
    float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_in_src);

    HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(), d->output_bitmap,
                            d->bitmap->image_size(), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(d->stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(d->stop));
    float times = 0;
    HANDLE_ERROR(cudaEventElapsedTime(&times, d->start, d->stop));

    d->total_time += times;
    ++d->frames;
    printf("Average time per frame: %3.1f ms\n", d->total_time / d->frames);
}

void anim_exit(DataBlock *d) {
    HANDLE_ERROR(cudaDestroyTextureObject(d->const_tex));
    cudaFree(d->dev_in_src);
    cudaFree(d->dev_out_src);
    cudaFree(d->dev_const_src);
    cudaFree(d->output_bitmap);

    HANDLE_ERROR(cudaEventDestroy(d->start));
    HANDLE_ERROR(cudaEventDestroy(d->stop));
}

int main(void) {
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;
    data.total_time = 0;
    data.frames = 0;
    data.const_tex = 0;

    HANDLE_ERROR(cudaEventCreate(&data.start));
    HANDLE_ERROR(cudaEventCreate(&data.stop));

    const size_t field_size = DIM * DIM * sizeof(float);

    HANDLE_ERROR(cudaMalloc((void **)&data.output_bitmap, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void **)&data.dev_in_src, field_size));
    HANDLE_ERROR(cudaMalloc((void **)&data.dev_out_src, field_size));
    HANDLE_ERROR(cudaMalloc((void **)&data.dev_const_src, field_size));

    float *temp = (float *)malloc(field_size);
    for (int i = 0; i < DIM * DIM; i++) {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x > 300) && (x < 600) && (y > 310) && (y < 601)) temp[i] = MAX_TEMP;
    }

    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
    temp[DIM * 700 + 100] = MIN_TEMP;
    temp[DIM * 300 + 300] = MIN_TEMP;
    temp[DIM * 200 + 700] = MIN_TEMP;

    for (int y = 800; y < 900; y++) {
        for (int x = 400; x < 500; x++) {
            temp[x + y * DIM] = MIN_TEMP;
        }
    }

    HANDLE_ERROR(cudaMemcpy(data.dev_const_src, temp, field_size, cudaMemcpyHostToDevice));

    for (int y = 800; y < DIM; y++) {
        for (int x = 0; x < 200; x++) {
            temp[x + y * DIM] = MAX_TEMP;
        }
    }

    HANDLE_ERROR(cudaMemcpy(data.dev_in_src, temp, field_size, cudaMemcpyHostToDevice));
    free(temp);

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr = data.dev_const_src;
    res_desc.res.linear.desc = cudaCreateChannelDesc<float>();
    res_desc.res.linear.sizeInBytes = field_size;

    cudaTextureDesc tex_desc = {};
    tex_desc.readMode = cudaReadModeElementType;

    HANDLE_ERROR(cudaCreateTextureObject(&data.const_tex, &res_desc, &tex_desc, nullptr));

    bitmap.anim_and_exit((void (*)(void *, int))anim_gpu,
                         (void (*)(void *))anim_exit);

    return 0;
}
