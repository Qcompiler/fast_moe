source /home/spack/spack/share/spack/setup-env.sh
spack load cuda@12.8
spack load cmake
spack load unzip
export PYTHONPATH=/home/chenyidong/newstart/bandwidth/jitcu
export PYTHONPATH=$PYTHONPATH:/home/chenyidong/newstart/bandwidth
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chenyidong/sgtest/w4a16kernel/kernel/build
