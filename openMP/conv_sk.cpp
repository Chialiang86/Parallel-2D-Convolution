#include <stdio.h>
#include <vector>
#include <sys/time.h>
#include <omp.h>
using namespace std;

vector<vector<float> > img;
vector<vector<float> > tmp_img;
vector<vector<float> > kernel;//row 0: vertical kernel, row 1: horizontal kernel
vector<vector<float> > ans;
int width,height,bpp,pad;
#include "../common/tools.hpp"

float cov_sk(int row,int col,int turn){
    float output=0;
    if(!turn){//turn==0
        for(int i=0;i<kernel[turn].size();i++){
            output+=kernel[turn][i]*img[row+i-pad][col];
        }
    }else{//turn==1
        for(int i=0;i<kernel[turn].size();i++){
            output+=kernel[turn][i]*tmp_img[row][col+i-pad];
        }
    }
    
    return output;
}

void inittmp_img(){
    vector<float> tmp;
    for(int i=pad;i<img.size();i++){
        tmp.clear();
        for(int j=pad;j<img[0].size();j++){
            tmp.push_back(0);
        }
        tmp_img.push_back(tmp);
    }
}

int main(int argc, char *argv[]){
    if(argc!=2){
        printf("Usage: ./conv_sk.out <# threads>\n");
        exit(0);
    }
    init_sk();
    inittmp_img();

    struct timeval start, end;
    long thread_cnt;
    gettimeofday(&start,0);
    thread_cnt = strtol(argv[1],NULL,10);

    for(int T = 0; T < RUN_NUM; T++){
        #pragma omp parallel for num_threads(thread_cnt)
        for(int i=pad;i<img.size()-pad;i++){
            for(int j=pad;j<img[0].size()-pad;j++){
                tmp_img[i][j]=cov_sk(i,j,0);
            }
        }
        #pragma omp parallel for num_threads(thread_cnt)
        for(int i=pad;i<img.size()-pad;i++){
            for(int j=pad;j<img[0].size()-pad;j++){
                ans[i-pad][j-pad]=cov_sk(i,j,1);
            }
        }
    }
    
    gettimeofday(&end, 0);
    int sec = end.tv_sec - start.tv_sec;
    int usec = end.tv_usec - start.tv_usec;
    printf("Elapsed time: %f sec\n", (sec+(usec/1000000.0))); 

    writeImage();
    checkAns();
    return 0;
}