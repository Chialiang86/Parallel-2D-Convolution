#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32

float *d_img, *d_ans, *d_kernel;

__global__ void conv(float *d_ans, float *d_img, float *d_kernel, 
                     int width, int height, int k_size, int pad) {

    int r = blockIdx.y * (blockDim.y - 2 * pad) + threadIdx.y;
    int c = blockIdx.x * (blockDim.x - 2 * pad) + threadIdx.x;
    int padded_img_width = width + 2 * pad;

    if (r >= height + 2 * pad || c >= width + 2 * pad)
        return;
    
    __shared__ float shared_patch[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_kernel[BLOCK_SIZE][BLOCK_SIZE];

    // tiling : load patch and kernel to shared memory in a SM
    shared_patch[threadIdx.y][threadIdx.x] = d_img[r * padded_img_width + c];

    shared_kernel[threadIdx.y % k_size][threadIdx.x % k_size] = d_kernel[(threadIdx.y % k_size) * k_size + threadIdx.x % k_size];
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
    d_ans[ans_r * width + ans_c] = res;
}

void mallocKernelAndAns(float *kernel_arr, int width, int height, int k_size, int pad) {

    cudaMalloc((void **)&d_img, (width + 2 * pad) * (height + 2 * pad) * sizeof(float));
    cudaMalloc((void **)&d_ans, width * height * sizeof(float));
    cudaMalloc((void **)&d_kernel, k_size * k_size * sizeof(float));
    cudaMemcpy(d_kernel, kernel_arr, k_size * k_size * sizeof(float), cudaMemcpyHostToDevice);

}

void convolution(float *img_arr, 
                 float *ans_arr,
                 int width, 
                 int height, 
                 int k_size, 
                 int pad) {

    // init cuda arr
    cudaMemcpy(d_img, img_arr, (width + 2 * pad) * (height + 2 * pad) * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlock(width / (BLOCK_SIZE - 2 * pad) + 1, height / (BLOCK_SIZE - 2 * pad) + 1);
    conv<<<numBlock, blockSize>>>(d_ans, d_img, d_kernel, width, height, k_size, pad);

    cudaMemcpy(ans_arr, d_ans, width * height * sizeof(float), cudaMemcpyDeviceToHost);

}

void freeKernelAndAns() {
    cudaFree(d_img);
    cudaFree(d_ans);
    cudaFree(d_kernel);
}