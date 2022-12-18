#ifndef KERNEL_H_
#define KERNEL_H_

//extern "C"
void mallocKernelAndAns(float *kernel_arr, int width, int height, int k_size, int pad);
void convolution(float *img_arr, 
                 float *ans_arr,
                 int width, 
                 int height, 
                 int k_size, 
                 int pad);
void freeKernelAndAns();

#endif /* KERNEL_H_ */
