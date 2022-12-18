#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include "kernel.h"

using namespace std;

vector<vector<float> > img;
vector<vector<float> > kernel;
vector<vector<float> > ans;
int width,height,bpp,pad;
#include "../common/tools.hpp"

float* vec2arr(const vector<vector<float> >& vec) {
    int rows = vec.size();
    int cols = vec[0].size();
    float *ret = new float[rows * cols];
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            ret[r * cols + c] = vec[r][c];
        }
    }
    return ret;
}

void arr2vec(vector<vector<float> >& vec, float *arr) {
    int rows = vec.size();
    int cols = vec[0].size();
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            vec[r][c] = arr[r * cols + c];
        }
    }
}

void display(float* arr, int width, int height) {
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            printf("%f ", arr[r * width + c]);
        }
        printf("\n");
    }
}

void display(const vector<vector<float> >& vec) {
    int rows = vec.size();
    int cols = vec[0].size();
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            printf("%f ", vec[r][c]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]){

    if (argc > 2)
        init_sk(argv[1], argv[2]);
    else if (argc == 2)
        init_sk(argv[1], "../common/kernel/kernel3x3_sk.txt");
    else 
        init_sk("../common/image/image.jpeg", "../common/kernel/kernel3x3_sk.txt");

    float *img_arr, *ans_arr, *kernel_arr;
    int k_size = kernel[0].size();

    img_arr = vec2arr(img);
    ans_arr = vec2arr(ans);
    kernel_arr = vec2arr(kernel);

    struct timeval start, end;
    gettimeofday(&start, 0);


    // init cuda mem
    mallocKernelAndAns(kernel_arr, width, height, k_size, pad);
    for(int T = 0; T < RUN_NUM; T++) {
        convolution(img_arr, ans_arr, width, height, k_size, pad);
    }
    freeKernelAndAns();

    gettimeofday(&end, 0);

    int sec = end.tv_sec - start.tv_sec;
    int usec = end.tv_usec - start.tv_usec;
    printf("Elapsed time: %f sec\n", (sec + (usec / 1000000.0))); 

    arr2vec(ans, ans_arr);

    char *out_txt_name, *out_img_name;
    out_txt_name = new char[256];
    out_img_name = new char[256];
    sprintf(out_txt_name, "./cuda_sk_%s", "ans.txt");
    sprintf(out_img_name, "./cuda_sk%s", ".jpeg");
    
    writeAns(out_txt_name);
    writeImage(out_img_name);
    checkAns("../serial/serial_sk_ans.txt");

    free(img_arr);
    free(ans_arr);
    free(kernel_arr);
    return 0;
}