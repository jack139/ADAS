# TEST


## 测试环境

- 基准测试 agent 使用 Qwen2.5-7B-Instruct

- Meta agent 使用 Qwen2.5-Coder-7B-Instruct



## 启动测试

```bash

# port 8000
llama-server -m qwen2.5-7b-instruct-q5_k_m.gguf -fa -c 10240 -ub 2048 -t 8 --port 8000 --n-gpu-layers 15000

# port 8001
llama-server -m qwen2.5-coder-7b-instruct-q5_k_m.gguf -fa -c 10240 -ub 2048 -t 8 --port 8001 --n-gpu-layers 15000

python3.9 _mgsm/search.py
```
