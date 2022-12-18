#include <stdio.h>
#include <vector>
#include <sys/time.h>
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

int main(int argc, char *argv[]){

    if (argc > 2)
        init(argv[1], argv[2]);
    else if (argc == 2)
        init(argv[1], "../common/kernel/kernel3x3.txt");
    else 
        init("../common/image/image.jpeg", "../common/kernel/kernel3x3.txt");

    struct timeval start, end;
    gettimeofday(&start, 0);

    for(int T = 0; T < RUN_NUM; T++){
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

    char *out_txt_name, *out_img_name;
    out_txt_name = new char[256];
    out_img_name = new char[256];
    sprintf(out_txt_name, "./serial_%s", "ans.txt");
    sprintf(out_img_name, "./image%s", ".jpeg");
    
    writeAns(out_txt_name);
    writeImage(out_img_name);
    return 0;
}