import os, sys, subprocess, time

PYTHON = sys.executable
TORCH_LIB = os.path.join(os.path.dirname(sys.executable), "..", "..", "Roaming", "Python", "Python311", "site-packages", "torch", "lib")
PROJECT = r"d:\trunk\SIG\output\cognitive-outsourcing"
MODELS = {
    "0.8B": os.path.join(PROJECT, "models", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    "4B": os.path.join(PROJECT, "models", "Qwen3.5-4B-Q4_K_M.gguf"),
}
OUTPUT = os.path.join(PROJECT, "_test_output.txt")

os.chdir(PROJECT)

def run(script, args, model_label=None):
    cmd = [PYTHON, "-u", script] + args
    env = os.environ.copy()
    env["PATH"] = TORCH_LIB + ";" + env.get("PATH", "")
    
    label = f"[{model_label}] " if model_label else ""
    print(f"\n{'='*70}")
    print(f"  RUNNING: {label}{script} {' '.join(args)}")
    print(f"{'='*70}\n", flush=True)
    
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output_lines = []
    for line in proc.stdout:
        print(line, end='', flush=True)
        output_lines.append(line)
    proc.wait()
    return proc.returncode, ''.join(output_lines)

full_output = []

# ===== CO Benchmark Tests =====
for model_name, model_path in MODELS.items():
    base = ["--n-ctx", "16384", "--n-gpu-layers", "99", "--no-debug"]
    
    for task in ["r6", "r13", "r14"]:
        rc, out = run("co_benchmark.py", ["--task", task, "--model", model_path] + base, model_name)
        full_output.append(f"# {model_name} co_benchmark --task {task}\n{out}\n")
        if task == "r13":
            time.sleep(1)  # R13 does eval on full cache, let GPU cool

# ===== SIG Benchmark Tests =====
for model_name, model_path in MODELS.items():
    base = ["--n-ctx", "16384", "--n-gpu-layers", "99", "--no-debug"]
    
    for task in ["r7", "r8", "r9"]:
        rc, out = run("sig_benchmark.py", ["--task", task, "--model", model_path] + base, model_name)
        full_output.append(f"# {model_name} sig_benchmark --task {task}\n{out}\n")
        time.sleep(1)

# ===== Transformer Tests: R10/R11/R12 with GGUF models =====
for model_name, model_path in MODELS.items():
    base = ["--n-ctx", "16384", "--n-gpu-layers", "99"]
    
    for task in ["r10", "r11", "r12"]:
        # transformer_bench R10/R11/R12 accept compiler=None gracefully
        rc, out = run("transformer_bench.py", ["--task", task, "--model-id", model_path], model_name)
        full_output.append(f"# {model_name} transformer_bench --task {task}\n{out}\n")

# ===== Transformer R1: Modelscope full-precision =====
for model_id in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]:
    rc, out = run("transformer_bench.py", ["--task", "r1", "--model-id", model_id], model_id)
    full_output.append(f"# R1 modelscope {model_id}\n{out}\n")

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write('\n'.join(full_output))

print(f"\n\nAll tests complete. Output saved to: {OUTPUT}")
