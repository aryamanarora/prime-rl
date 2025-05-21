from datasets import load_dataset

data = load_dataset("PrimeIntellect/INTELLECT-2-RL-Dataset")["train"]

data = data.filter(lambda x: x["task_type"] == "verifiable_math")
data1 = data.filter(lambda x: x["solve_rate_qwen_r1_distill_7b"] < 0.51 and x["solve_rate_qwen_r1_distill_7b"] > 0.03)


print("len", len(data1))

#data1.push_to_hub("justus27/rl-math-filtered-difficult")