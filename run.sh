# bin/sh

cd serial/
make
./conv.out
./conv_sk.out
cd ..

cd pthread/
make
./pthread 2
./pthread 3
./pthread 4
./pthread 5
./pthread 6
./pthread 7
./pthread 8
cd ..

cd cuda/
make
./cuda_basic
./cuda_pitch
./cuda_tiling
./cuda_tiling+pitch
./cuda_sk_basic