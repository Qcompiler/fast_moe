source load.sh

conda create python==3.11 -n dsl
conda activate dsl



cd w4a16kernel/kernel/build
cmake ..
make -j
cd w4a16kernel/kernel
