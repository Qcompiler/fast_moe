source /home/spack/spack/share/spack/setup-env.sh
spack load cuda@12.8
spack load cmake
spack load unzip
export PYTHONPATH=/home/chenyidong/newstart/bandwidth/jitcu
export PYTHONPATH=$PYTHONPATH:/home/chenyidong/newstart/bandwidth
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chenyidong/sgtest/w4a16kernel/kernel/build


# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/spack/spack/opt/spack/linux-debian12-sapphirerapids/gcc-12.2.0/cuda-12.8.0-ogkmenn2commmbqjel5iws2ieekvevsj/lib64

conda activate dsl



# 为本地安装的软件添加环境变量
# export PATH="$HOME/my_software/usr/bin:$PATH"
# export LD_LIBRARY_PATH="$HOME/my_software/usr/lib/x86_64-linux-gnu:$HOME/my_software/usr/lib:$LD_LIBRARY_PATH"
# export XDG_DATA_DIRS="$HOME/my_software/usr/share:$XDG_DATA_DIRS"
# export PKG_CONFIG_PATH="$HOME/my_software/usr/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH"
