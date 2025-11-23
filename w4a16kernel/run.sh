


# srun -N 1 --cpus-per-task=32 --pty --gres=gpu:4090:1  python check_gate_up_output.py 
srun -N 1 --cpus-per-task=32 --pty --gres=gpu:H100:1  python check_down_output.py

# srun -N 1 --cpus-per-task=32 --pty --gres=gpu:H100:1  python test.py