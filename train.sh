# Start trainer
ulimit -n 4096
export CUDA_VISIBLE_DEVICES=6,7
uv  run torchrun --nproc_per_node=2 src/zeroband/train.py @ configs/training/rl_calibration_math.toml --data.num_workers 2