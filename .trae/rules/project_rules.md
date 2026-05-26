# Project Rules

## Python Environment

Use Python 3.11 with CUDA acceleration. **Must add torch/lib to DLL search path** before importing llama-cpp-python:

```powershell
$env:PATH = "<path-to-torch-lib>;" + $env:PATH
& python script.py ...
```

Or use the helper `run_test.py` wrapper:

```powershell
python run_test.py --task r6 --model models/Qwen3.5-0.8B-Q4_K_M.gguf
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
