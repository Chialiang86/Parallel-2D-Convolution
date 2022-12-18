#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define IMAGE_CACHE_WIDTH 48
#define KERNEL_CACHE_WIDTH 16
#define BLOCK_SIZE 32

float *d_img, *d_ans, *d_kernel;

__global__ void conv(float *d_ans, float *d_img, float *d_kernel, 
                     int width, int height, int k_size, int pad) {

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int padded_img_width = width + 2 * pad;
    int padded_img_height = height + 2 * pad;

    if (r >= height || c >= width)
        return;
    
    __shared__ float shared_patch[IMAGE_CACHE_WIDTH][IMAGE_CACHE_WIDTH];
    __shared__ float shared_kernel[KERNEL_CACHE_WIDTH][KERNEL_CACHE_WIDTH];

    // tiling : load patch and kernel to shared memory in a SM
    shared_patch[threadIdx.y + pad][threadIdx.x + pad] = d_img[r * padded_img_width + c];
    
    // left part
    if (threadIdx.x < pad && c >= pad)
        shared_patch[threadIdx.y + pad][threadIdx.x] = d_img[r * padded_img_width + c - pad];
    // right part
    if (threadIdx.x >= BLOCK_SIZE - pad && c < padded_img_width - pad)
        shared_patch[threadIdx.y + pad][threadIdx.x + pad + pad] = d_img[r * padded_img_width + c + pad];
    // top part
    if (threadIdx.y < pad && r >= pad)
        shared_patch[threadIdx.y][threadIdx.x + pad] = d_img[(r - pad) * padded_img_width + c];
    // bottom part
    if (threadIdx.y >= BLOCK_SIZE - pad && r < padded_img_height - pad)
        shared_patch[threadIdx.y + pad + pad][threadIdx.x + pad] = d_img[(r + pad) * padded_img_width + c];
    // top-left part
    if (threadIdx.y < pad && r >= pad && threadIdx.x < pad && c >= pad)
        shared_patch[threadIdx.y][threadIdx.x] = d_img[(r - pad) * padded_img_width + c - pad];
    // top-right part
    if (threadIdx.y < pad && r >= pad && threadIdx.x >= BLOCK_SIZE - pad && c < padded_img_width - pad)
        shared_patch[threadIdx.y][threadIdx.x + pad + pad] = d_img[(r - pad) * padded_img_width + c + pad];
    // bottom-left part
    if (threadIdx.y >= BLOCK_SIZE - pad && r < padded_img_height - pad && threadIdx.x < pad && c >= pad)
        shared_patch[threadIdx.y + pad + pad][threadIdx.x] = d_img[(r + pad) * padded_img_width + c - pad];
    // bottom-right part
    if (threadIdx.y >= BLOCK_SIZE - pad && r < padded_img_height - pad && threadIdx.x >= BLOCK_SIZE - pad && c < padded_img_width - pad)
        shared_patch[threadIdx.y + pad + pad][threadIdx.x + pad + pad] = d_img[(r + pad) * padded_img_width + c + pad];

    shared_kernel[threadIdx.y % k_size][threadIdx.x % k_size] = d_kernel[(threadIdx.y % k_size) * k_size + threadIdx.x % k_size];
    __syncthreads();

    if (r == 0 && c == 0) {
        for (int row = 0; row < BLOCK_SIZE + 2 * pad; row++) {
            for (int col = 0; col < BLOCK_SIZE + 2 * pad; col++) {
                printf("%f ", shared_patch[row][col] - d_img[row * padded_img_width + col]);
            }
            printf("\n");
        }
    }
    
    float res = 0.0;
    for (int kr = -pad; kr <= pad; kr++) {
        for (int kc = -pad; kc <= pad; kc++) {
            res += shared_patch[threadIdx.y + kr + pad][threadIdx.x + kc + pad] * shared_kernel[kr + pad][kc + pad];
        }
    }
    d_ans[r * width + c] = res;
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
    dim3 numBlock(width / BLOCK_SIZE + 1, height / BLOCK_SIZE + 1);
    conv<<<numBlock, blockSize>>>(d_ans, d_img, d_kernel, width, height, k_size, pad);

    cudaMemcpy(ans_arr, d_ans, width * height * sizeof(float), cudaMemcpyDeviceToHost);

}

void freeKernelAndAns() {
    cudaFree(d_img);
    cudaFree(d_ans);
    cudaFree(d_kernel);
}