docker run --rm --gpus all \
  -v /home/dataset/tmp:/home/chenyidong \
  -it --name rocm \
  docker.1ms.run/rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1 \
  bash

srun -N 1  --pty --gres=gpu:MI210:1 -p Long  --pty docker pull docker.1ms.run/rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1


alias drun='docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G -v /home/dataset/tmp:/dockerx -w /dockerx'
drun docker.1ms.run/rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1