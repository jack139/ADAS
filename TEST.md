# TEST


## 测试环境

- 基准测试 agent 使用 Qwen2.5-7B-Instruct

- Meta agent 使用 Qwen2.5-Coder-7B-Instruct



## 启动测试

```bash

# port 8000
llama-server -m qwen2.5-7b-instruct-q5_k_m.gguf -fa -ub 2048 -t 8 --port 8000 --n-gpu-layers 15000 --yarn-orig-ctx 32768 --yarn-ext-factor 4.0

# port 8001
llama-server -m qwen2.5-coder-7b-instruct-q5_k_m.gguf -fa -ub 2048 -t 8 --port 8001 --n-gpu-layers 15000 --yarn-orig-ctx 32768 --yarn-ext-factor 4.0

python3.9 _mgsm/search.py
```
