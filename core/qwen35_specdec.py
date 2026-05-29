"""Qwen3.5-compatible speculative decoding via eval()+sample().

This module implements the draft-verify loop using only eval() and
sample(), completely bypassing the broken generate() method.

The key insight: eval() + sample() work perfectly on Qwen3.5.
Only generate()'s kv_cache_seq_rm partial truncation is broken.

Architecture:
1. eval(context) → sample() → get target token
2. Run drafter on input_ids → get K draft tokens
3. eval(draft_tokens) → sample() at each position → verify
4. Accept consecutive matches, reject at first mismatch
5. On rejection: reset() + re-eval accepted prefix
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple


class Qwen35SpecDecCompiler:
    """Speculative decoding for Qwen3.5 using eval()+sample().

    Completely avoids generate() and its broken kv_cache_seq_rm.
    Uses reset() + re-eval for rejection handling.
    """

    def __init__(self, model_path: str, n_ctx: int = 4096,
                 n_gpu_layers: int = 99, draft_model=None):
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            logits_all=True,  # Required for draft verification
            draft_model=draft_model,
        )
        self.draft_model = draft_model

    def tokenize(self, text: str, add_bos: bool = False) -> List[int]:
        if isinstance(text, str):
            text = text.encode()
        return list(self.llm.tokenize(text, add_bos=add_bos))

    def detokenize(self, token_ids: List[int]) -> str:
        return self.llm.detokenize(token_ids).decode("utf-8", errors="replace")

    def reset(self):
        self.llm.reset()

    def eval(self, token_ids: List[int]):
        self.llm.eval(token_ids)

    def sample(self, idx: Optional[int] = None) -> int:
        if idx is not None:
            return self.llm.sample(idx=idx, temp=0.0)
        return self.llm.sample(temp=0.0)

    def _detect_repetition(self, text: str, threshold: int = 3) -> bool:
        if len(text) < 50:
            return False
        words = text.split()
        if len(words) < threshold * 3:
            return False
        for n in range(2, min(10, len(words) // threshold)):
            gram = " ".join(words[-n:])
            if text.count(gram) >= threshold:
                return True
        return False

    def generate_with_specdec(
        self,
        max_new: int = 60,
        stop_str: str = "\nUser:",
    ) -> Tuple[str, List[int], Dict]:
        """Generate tokens using speculative decoding.

        Must call eval(context_tokens) before calling this method.

        Returns: (text, gen_ids, stats)
        """
        gen_ids: List[int] = []
        stats = {
            "draft_proposed": 0,
            "draft_accepted": 0,
            "total_tokens": 0,
            "steps": 0,
            "rejections": 0,
        }

        while len(gen_ids) < max_new:
            stats["steps"] += 1

            # Step 1: Sample target token
            target_token = self.sample()

            # Step 2: Get draft predictions from drafter
            draft_tokens = []
            if self.draft_model is not None:
                input_ids = self.llm.input_ids[:self.llm.n_tokens].copy()
                draft_result = self.draft_model(input_ids)
                draft_tokens = list(draft_result.astype(int))

            if not draft_tokens or draft_tokens[0] != target_token:
                # No draft match: just eval the target token
                gen_ids.append(target_token)
                self.eval([target_token])
                if draft_tokens:
                    stats["draft_proposed"] += len(draft_tokens)
                # Check stop on the last few generated tokens
                if len(gen_ids) >= 5:
                    last_text = self.detokenize(gen_ids[-10:])
                    if stop_str in last_text:
                        break
                continue

            # Step 3: Draft[0] matches! Eval target + draft[1:] for verification
            stats["draft_proposed"] += len(draft_tokens)
            stats["draft_accepted"] += 1

            # Eval all draft tokens (including the matched first one)
            self.eval(draft_tokens)

            # Step 4: Verify each draft token
            accepted = [target_token]
            for i in range(1, len(draft_tokens)):
                verify_idx = self.llm.n_tokens - len(draft_tokens) + i
                verified_token = self.sample(idx=verify_idx)

                if verified_token == draft_tokens[i]:
                    accepted.append(draft_tokens[i])
                    stats["draft_accepted"] += 1
                else:
                    # Rejection! Reset and re-eval accepted prefix
                    stats["rejections"] += 1
                    accepted_prefix = list(self.llm.input_ids[:self.llm.n_tokens - len(draft_tokens) + i])
                    accepted_prefix.append(verified_token)
                    self.reset()
                    self.eval(accepted_prefix)
                    gen_ids.extend(accepted)
                    gen_ids.append(verified_token)
                    self.eval([verified_token])
                    break
            else:
                # All draft tokens accepted!
                gen_ids.extend(draft_tokens)

            stats["total_tokens"] = len(gen_ids)

            # Check stop on last generated tokens
            if len(gen_ids) >= 5:
                last_text = self.detokenize(gen_ids[-10:])
                if stop_str in last_text:
                    break
            if self._detect_repetition(self.detokenize(gen_ids)):
                break

        text = self.detokenize(gen_ids)
        if stop_str in text:
            text = text[:text.index(stop_str)]
        return text, gen_ids, stats

    def generate_simple(self, max_new: int = 60, stop_str: str = "\nUser:") -> Tuple[str, List[int]]:
        """Simple generation without speculation (for baseline)."""
        gen_ids: List[int] = []

        while len(gen_ids) < max_new:
            tok = self.sample()
            gen_ids.append(tok)
            self.eval([tok])

            if len(gen_ids) >= 5:
                last_text = self.detokenize(gen_ids[-10:])
                if stop_str in last_text:
                    break
            if self._detect_repetition(self.detokenize(gen_ids)):
                break

        text = self.detokenize(gen_ids)
        if stop_str in text:
            text = text[:text.index(stop_str)]
        return text, gen_ids
