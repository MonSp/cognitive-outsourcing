"""Manual speculative decoding for llama-cpp-python with Qwen3.5.

llama.cpp's native Llama.generate() + drafter crashes on Qwen3.5
(llama_decode returned -1). This module implements the draft-verify
loop manually using eval() + sample(), which work correctly.

The drafter uses n-gram matching on the token history — no separate
model needed, zero VRAM overhead.
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple


class NgramDrafter:
    """N-gram drafter for manual speculative decoding.

    Searches the token history for matching n-gram patterns and
    predicts the continuation tokens.
    """

    def __init__(self, ngram_size: int = 3, num_pred: int = 3):
        self.ngram_size = ngram_size
        self.num_pred = num_pred
        self.history: List[int] = []

    def reset(self):
        self.history = []

    def add_tokens(self, tokens: List[int]):
        self.history.extend(tokens)

    def predict(self) -> List[int]:
        """Predict next tokens using n-gram matching on history."""
        if len(self.history) < self.ngram_size + 1:
            return []

        ngram = self.history[-self.ngram_size:]
        for i in range(len(self.history) - self.ngram_size - 1, -1, -1):
            if self.history[i:i + self.ngram_size] == ngram:
                start = i + self.ngram_size
                end = min(start + self.num_pred, len(self.history))
                if start < end:
                    return self.history[start:end]
        return []


class ManualSpecDecCompiler:
    """Manual speculative decoding using eval() + sample().

    Uses n-gram drafting + sequential verification. Each verification
    step only processes 1 token (via eval).

    Note: Sequential verification has the same per-token cost as normal
    generation (1 forward pass per token). True speedup requires parallel
    verification (eval K tokens in 1 forward pass), which needs
    logits_all=True + kv_cache_seq_rm support. On Qwen3.5, both
    logits_all=True (sample(idx=...) crashes) and kv_cache_seq_rm
    (partial deletion fails) are broken due to hybrid attention.
    """

    def __init__(self, model_path: str, n_ctx: int = 16384,
                 n_gpu_layers: int = 99, ngram_size: int = 3,
                 num_pred: int = 3):
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.drafter = NgramDrafter(ngram_size=ngram_size, num_pred=num_pred)
        self.n_tokens = 0

    def tokenize(self, text: str, add_bos: bool = False) -> List[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos))

    def detokenize(self, token_ids: List[int]) -> str:
        return self.llm.detokenize(token_ids).decode("utf-8", errors="replace")

    def reset_cache(self):
        self.llm._ctx.kv_cache_clear()
        self.llm.n_tokens = 0
        self.n_tokens = 0
        self.drafter.reset()

    def eval(self, token_ids: List[int]):
        self.llm.eval(token_ids)
        self.n_tokens += len(token_ids)
        self.drafter.add_tokens(token_ids)

    def sample(self) -> int:
        return self.llm.sample(temp=0.0)

    def generate_with_specdec(
        self,
        stop_str: str = "\nUser:",
        max_new: int = 60,
        rep_threshold: int = 3,
        min_tokens_before_stop: int = 5,
        num_pred: int = 3,
    ) -> Tuple[str, List[int], Dict]:
        """Generate tokens using manual speculative decoding.

        Returns: (text, gen_ids, stats)
        """
        gen_ids: List[int] = []
        stop_strs = [stop_str, "Assistant:", "assistant:"]
        stats = {
            "draft_proposed": 0,
            "draft_accepted": 0,
            "total_tokens": 0,
            "steps": 0,
        }

        while len(gen_ids) < max_new:
            stats["steps"] += 1

            # Step 1: Get target model's prediction
            target_token = self.sample()

            # Step 2: Get drafter's predictions
            draft_tokens = self.drafter.predict()[:num_pred]

            if draft_tokens and draft_tokens[0] == target_token:
                # Draft[0] matches target — accept and verify further
                accepted_tokens = [target_token]
                self.eval([target_token])
                stats["draft_proposed"] += len(draft_tokens)
                stats["draft_accepted"] += 1

                # Verify remaining draft tokens
                for i in range(1, len(draft_tokens)):
                    next_target = self.sample()
                    if draft_tokens[i] == next_target:
                        accepted_tokens.append(draft_tokens[i])
                        self.eval([draft_tokens[i]])
                        stats["draft_accepted"] += 1
                    else:
                        # Reject: use target's token instead
                        accepted_tokens.append(next_target)
                        self.eval([next_target])
                        break

                gen_ids.extend(accepted_tokens)
            else:
                # No draft match: just use target token
                gen_ids.append(target_token)
                self.eval([target_token])
                if draft_tokens:
                    stats["draft_proposed"] += len(draft_tokens)

            stats["total_tokens"] = len(gen_ids)

            # Check stop conditions
            cur = self.detokenize(gen_ids)
            if len(gen_ids) >= min_tokens_before_stop:
                for s in stop_strs:
                    if s in cur:
                        return cur.split(s)[0], gen_ids, stats

            if self._detect_repetition(cur, threshold=rep_threshold):
                break

        return self.detokenize(gen_ids), gen_ids, stats

    def _detect_repetition(self, text: str, threshold: int = 3) -> bool:
        if len(text) < 50:
            return False
        words = text.split()
        if len(words) < threshold * 3:
            return False
        for n in range(2, min(10, len(words) // threshold)):
            gram = " ".join(words[-n:])
            count = text.count(gram)
            if count >= threshold:
                return True
        return False


def run_benchmark(model_path: str, prompt: str, max_new: int = 50,
                  n_gpu_layers: int = 99, num_pred: int = 3):
    """Benchmark manual SpecDec vs baseline."""
    from llama_cpp import Llama

    # Baseline: standard eval+sample loop
    print("Baseline (eval+sample)...")
    llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=n_gpu_layers, verbose=False)
    tokens = list(llm.tokenize(prompt.encode()))
    llm.eval(tokens)
    gen_ids = []
    t0 = time.time()
    for _ in range(max_new):
        tok = llm.sample(temp=0.0)
        gen_ids.append(tok)
        llm.eval([tok])
        text = llm.detokenize(gen_ids).decode("utf-8", errors="replace")
        if "\nUser:" in text or "Assistant:" in text[len(text)//2:]:
            break
    baseline_time = time.time() - t0
    baseline_text = llm.detokenize(gen_ids).decode("utf-8", errors="replace")
    print(f"  {len(gen_ids)} tokens in {baseline_time:.2f}s ({len(gen_ids)/baseline_time:.1f} tok/s)")
    del llm

    # SpecDec: manual draft-verify loop
    print(f"SpecDec (ngram, k={num_pred})...")
    compiler = ManualSpecDecCompiler(
        model_path=model_path, n_ctx=4096, n_gpu_layers=n_gpu_layers,
        ngram_size=3, num_pred=num_pred)
    tokens = compiler.tokenize(prompt)
    compiler.reset_cache()
    compiler.eval(tokens)
    t0 = time.time()
    text, gen_ids, stats = compiler.generate_with_specdec(
        stop_str="\nUser:", max_new=max_new, num_pred=num_pred)
    specdec_time = time.time() - t0
    print(f"  {len(gen_ids)} tokens in {specdec_time:.2f}s ({len(gen_ids)/specdec_time:.1f} tok/s)")
    print(f"  Draft proposed: {stats['draft_proposed']}, accepted: {stats['draft_accepted']}")
    if stats['draft_proposed'] > 0:
        print(f"  Acceptance rate: {100*stats['draft_accepted']/stats['draft_proposed']:.1f}%")
    print(f"  Speedup: {baseline_time/specdec_time:.2f}x")

    return {
        "baseline": {"time": baseline_time, "tokens": len(gen_ids)},
        "specdec": {"time": specdec_time, "tokens": len(gen_ids), "stats": stats},
        "speedup": baseline_time / specdec_time if specdec_time > 0 else 0,
    }


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3.5-4B-Q4_K_M.gguf"
    prompt = "You are a kitchen assistant.\n\nUser: What recipes can I make with chicken?\nAssistant:"
    run_benchmark(model, prompt, max_new=50, num_pred=3)
