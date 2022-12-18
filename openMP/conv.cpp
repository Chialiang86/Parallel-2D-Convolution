#include <stdio.h>
#include <vector>
#include <sys/time.h>
#include <omp.h>
using namespace std;

vector<vector<float> > img;
vector<vector<float> > kernel;
vector<vector<float> > ans;
int width,height,bpp,pad;
#include "../common/tools.hpp"

float cov(int row,int col){
    float output=0;
    for(int i=0;i<kernel.size();i++){
        for(int j=0;j<kernel[0].size();j++){
            output+=kernel[i][j]*img[row+i-pad][col+j-pad];
        }
    }
    return output;
}

int main(int argc,char *argv[]){
    if(argc!=2){
        printf("Usage: ./conv.out <# threads>\n");
        exit(0);
    }
    init();

    struct timeval start, end;
    long thread_cnt;

    gettimeofday(&start,0);
    thread_cnt=strtol(argv[1],NULL,10);
    for(int T = 0; T < RUN_NUM; T++){
        #pragma omp parallel for num_threads(thread_cnt)
        for(int i=pad;i<img.size()-pad;i++){
            for(int j=pad;j<img[0].size()-pad;j++){
                ans[i-pad][j-pad]=cov(i,j);
            }
        }
    }
    
    gettimeofday(&end, 0);
    int sec = end.tv_sec - start.tv_sec;
    int usec = end.tv_usec - start.tv_usec;
    printf("Elapsed time: %f sec\n", (sec+(usec/1000000.0))); 

    checkAns();
    return 0;
}