#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 42

float *d_img, *d_ans, *d_kernel;
size_t img_pitch, ans_pitch, kernel_pitch;

__global__ void conv(float *d_ans, float *d_img, float *d_kernel,
                    size_t ans_pitch, size_t img_pitch, size_t kernel_pitch,
                    int width, int height, int k_size, int pad) {

    int r = blockIdx.y * (blockDim.y - 2 * pad) + threadIdx.y;
    int c = blockIdx.x * (blockDim.x - 2 * pad) + threadIdx.x;

    if (r >= height + 2 * pad || c >= width + 2 * pad)
        return;
    
    __shared__ float shared_patch[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_kernel[BLOCK_SIZE][BLOCK_SIZE];

    // tiling : load patch and kernel to shared memory in a SM
    shared_patch[threadIdx.y][threadIdx.x] = 
        *((float*)((char*)d_img + r * img_pitch) + c);
    shared_kernel[threadIdx.y % k_size][threadIdx.x % k_size] = 
        *((float*)((char*)d_kernel + (threadIdx.y % k_size) * kernel_pitch) + threadIdx.x % k_size);
    __syncthreads();

    if (threadIdx.y < pad || threadIdx.y >= BLOCK_SIZE - pad || 
        threadIdx.x < pad || threadIdx.x >= BLOCK_SIZE - pad ||
        r >= height + pad || c >= width + pad)
        return;

    int ans_r = r - pad; 
    int ans_c = c - pad; 
    
    float res = 0.0;
    for (int kr = -pad; kr <= pad; kr++) {
        for (int kc = -pad; kc <= pad; kc++) {
            res += shared_patch[threadIdx.y + kr][threadIdx.x + kc] * shared_kernel[kr + pad][kc + pad];
        }
    }

    *((float*)((char*)d_ans + ans_r * ans_pitch) + ans_c) = res;
}

void mallocKernelAndAns(float *kernel_arr, int width, int height, int k_size, int pad) {

    cudaMallocPitch((void **)&d_img, &img_pitch, (width + 2 * pad) * sizeof(float), height + 2 * pad);
    cudaMallocPitch((void **)&d_ans, &ans_pitch, width * sizeof(float), height);
    cudaMallocPitch((void **)&d_kernel, &kernel_pitch, k_size * sizeof(float), k_size);
    cudaMemcpy2D(d_kernel, kernel_pitch, 
                    kernel_arr, k_size * sizeof(float), 
                    k_size * sizeof(float), k_size, 
                    cudaMemcpyHostToDevice);
}

void convolution(float *img_arr, 
                 float *ans_arr,
                 int width, 
                 int height, 
                 int k_size, 
                 int pad) {

    // init cuda arr
    cudaMemcpy2D(d_img, img_pitch, 
                img_arr, (width + 2 * pad) * sizeof(float), 
                (width + 2 * pad) * sizeof(float), (height + 2 * pad), 
                cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlock(width / (BLOCK_SIZE - 2 * pad) + 1, height / (BLOCK_SIZE - 2 * pad) + 1);
    conv<<<numBlock, blockSize>>>(d_ans, d_img, d_kernel, 
                                 ans_pitch, img_pitch, kernel_pitch,
                                 width, height, k_size, pad);
    
    cudaMemcpy2D(ans_arr, width * sizeof(float), 
                d_ans, ans_pitch, 
                width * sizeof(float), height, 
                cudaMemcpyDeviceToHost);
    
}

void freeKernelAndAns() {
    cudaFree(d_img);
    cudaFree(d_ans);
    cudaFree(d_kernel);
}