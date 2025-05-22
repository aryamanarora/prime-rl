export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export VLLM_WORKER_MULTIPROC_METHOD=spawn
uv run python src/zeroband/infer.py @ configs/inference/rl_calibration_math.toml