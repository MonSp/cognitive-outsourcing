# Project Rules

## Python Environment

Use conda environment `sig_bench` at `D:\ProgramData\miniconda3\envs\sig_bench`:

```powershell
& "D:\ProgramData\miniconda3\envs\sig_bench\python.exe" script.py ...
```

Or activate the environment first:

```powershell
conda activate sig_bench
python script.py ...
```

All scripts require:
- `llama-cpp-python` — GGUF model inference (CUDA)
- `torch` 2.6.0+cu124, `transformers`, `accelerate` — HuggingFace transformer models
- `modelscope` — ModelScope model hub
- `numpy` — numerical computation
- `pynvml` — GPU monitoring

## Models

GGUF models for llama.cpp tests:
- `models/Qwen3.5-0.8B-Q4_K_M.gguf` — 0.8B quantized
- `models/Qwen3.5-4B-Q4_K_M.gguf` — 4B quantized

For transformer_bench (R1 attention analysis via modelscope):
- `Qwen/Qwen2.5-0.5B` (default)
- `Qwen/Qwen2.5-1.5B`
