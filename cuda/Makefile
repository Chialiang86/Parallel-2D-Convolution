NVCC = nvcc
CXX = g++
CXXFLAGS=-I./common -Iobjs/ -O3 -std=c++17 -Wall -g -fPIC -lm
CXXINCS = -I ../common

# CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' 
# CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3

EXE = cuda_basic cuda_pitch cuda_tiling cuda_tiling+pitch cuda_sk_basic
CPP = pthread.cpp

OBJS = cuda.o cuda_sk.o kernel_basic.o kernel_pitch.o kernel_tiling.o kernel_tiling+pitch.o kernel_sk_basic.o

all: $(EXE)
	rm $(OBJS)

# for general kernel
cuda_basic: kernel_basic.o cuda.o
	$(NVCC) -o $@ $^

cuda_pitch: kernel_pitch.o cuda.o
	$(NVCC) -o $@ $^

cuda_tiling: kernel_tiling.o cuda.o
	$(NVCC) -o $@ $^

cuda_tiling+pitch: kernel_tiling+pitch.o cuda.o
	$(NVCC) -o $@ $^

kernel_basic.o: kernel_basic.cu
	$(NVCC) -c $< -o $@

kernel_pitch.o: kernel_pitch.cu
	$(NVCC) -c $< -o $@

kernel_tiling.o: kernel_tiling.cu
	$(NVCC) -c $< -o $@

kernel_tiling+pitch.o: kernel_tiling+pitch.cu
	$(NVCC) -c $< -o $@

# for sk kernel
cuda_sk_basic: kernel_sk_basic.o cuda_sk.o
	$(NVCC) -o $@ $^

kernel_sk_basic.o: kernel_sk_basic.cu
	$(NVCC) -c $< -o $@

# for cpp files
cuda.o: cuda.cpp
	$(CXX) -c $< $(CXXFLAGS) $(CXXINCS) -o $@

cuda_sk.o: cuda_sk.cpp
	$(CXX) -c $< $(CXXFLAGS) $(CXXINCS) -o $@

.PHONY: clean

clean:                             
	rm -rf $(EXE) $(OBJS)