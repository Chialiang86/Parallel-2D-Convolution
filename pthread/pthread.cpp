#include <stdint.h>
#include <pthread.h>
#include <stdio.h>
#include <vector>
#include <sys/time.h>
using namespace std;

// image variables
vector<vector<float> > img;
vector<vector<float> > kernel;
vector<vector<float> > ans;
int width, height, bpp, pad;
#include "../common/tools.hpp"

// shared variables
long thread_cnt;

void compute_conv(long row_id, vector<vector<float> > &ans){
    for(int col_id = pad; col_id < width + pad; col_id++){
        float out = 0;

        for(int i = 0; i < kernel.size(); i++){
            for(int j = 0; j < kernel[0].size(); j++){
                out += (kernel[i][j] * img[row_id + i - pad][col_id + j - pad]);
            }
        }

        ans[row_id - pad][col_id - pad] = out;
    }
}

void* thread_conv_func(void *rank){
    long my_rank = (long) rank;

    for(long row_id = pad + my_rank; row_id <= height; row_id += thread_cnt){
        compute_conv(row_id, ans);
    }

    return NULL;
}

// void* thread_conv_func(void *rank){
//     long my_rank = (long) rank;
//     int step = (height + pad) / thread_cnt;
//     long long int my_first_i = step * my_rank;
//     long long int my_last_i = (my_rank == thread_cnt - 1) ? height + 1 : my_first_i + step;

//     for(long row_id = pad + my_first_i; row_id < my_last_i; row_id++){
//         for(int col_id = pad; col_id < width + pad; col_id++){
//             float out = 0;

//             for(int i = 0; i < kernel.size(); i++){
//                 for(int j = 0; j < kernel[0].size(); j++){
//                     out += (kernel[i][j] * img[row_id + i - pad][col_id + j - pad]);
//                 }
//             }

//             ans[row_id - pad][col_id - pad] = out;
//         }
//     }

//     return NULL;
// }

int main(int argc, char *argv[]){

        
    pthread_t* thread_handles;
    struct timeval start, end;

    if(argc != 2){
        printf("Usage: ./pthread <# threads>\n");
        exit(0);
    }

    if (argc > 3)
        init(argv[2], argv[3]);
    else if (argc == 3)
        init(argv[2], "../common/kernel/kernel3x3.txt");
    else 
        init("../common/image/image.jpeg", "../common/kernel/kernel3x3.txt");

    gettimeofday(&start, 0);

    thread_cnt = strtol(argv[1], NULL, 10);
    thread_handles = (pthread_t*) malloc (thread_cnt * sizeof(pthread_t));

    for(int T = 0; T < RUN_NUM; T++){
        // create threads
        for(long thread = 0; thread < thread_cnt; thread++)
            pthread_create(&thread_handles[thread], NULL, thread_conv_func, (void*) thread);
        
        // wait for threads join
        for(long thread = 0; thread < thread_cnt; thread++)
            pthread_join(thread_handles[thread], NULL);
    }

    gettimeofday(&end, 0);
    int sec = end.tv_sec - start.tv_sec;
    int usec = end.tv_usec - start.tv_usec;
    printf("Elapsed time: %f sec\n", (sec+(usec/1000000.0))); 

    char *out_txt_name, *out_img_name;
    out_txt_name = new char[256];
    out_img_name = new char[256];
    sprintf(out_txt_name, "./pthread_%s", "ans.txt");
    sprintf(out_img_name, "./pthread%s", ".jpeg");
    
    writeAns(out_txt_name);
    writeImage(out_img_name);
    checkAns("../serial/serial_ans.txt");

    return 0;
}