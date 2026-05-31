"""Qwen3.5-compatible speculative decoding via eval()+sample().

This module implements the draft-verify loop using only eval() and
sample(), completely bypassing the broken generate() method.

The key insight: eval() + sample() work perfectly on Qwen3.5.
Only generate()'s kv_cache_seq_rm partial truncation is broken.

Architecture:
1. eval(context) -> sample() -> get target token
2. Run drafter via eval()+sample() -> get K draft tokens
3. eval(draft_tokens) -> sample() at each position -> verify
4. Accept consecutive matches, reject at first mismatch
5. On rejection: undo draft tokens on drafter, re-eval on target
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple


class Qwen35SpecDecCompiler:
    """Speculative decoding for Qwen3.5 using eval()+sample().

    Completely avoids generate() and its broken kv_cache_seq_rm.
    Uses eval()+sample() on both target and drafter models.

    The drafter model generates K candidate tokens autoregressively.
    The target model verifies each token. On mismatch, the target
    resets and re-evals the accepted prefix. The drafter's cache is
    rolled back using kv_cache_seq_rm (if supported) or reset+re-eval.
    """

    def __init__(self, model_path: str, n_ctx: int = 16384,
                 n_gpu_layers: int = 99, drafter_path: Optional[str] = None):
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        self.drafter_llm = None
        if drafter_path is not None:
            self.drafter_llm = Llama(
                model_path=drafter_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )

        self._drafter_supports_partial_rm = None

    def _test_drafter_partial_rm(self) -> bool:
        """Test if the drafter model supports kv_cache_seq_rm partial deletion."""
        if self._drafter_supports_partial_rm is not None:
            return self._drafter_supports_partial_rm
        if self.drafter_llm is None:
            self._drafter_supports_partial_rm = False
            return False
        try:
            drafter = self.drafter_llm
            drafter.reset()
            dummy = [1, 2, 3, 4, 5]
            drafter.eval(dummy)
            n_before = drafter.n_tokens
            result = drafter._ctx.kv_cache_seq_rm(0, 3, n_before)
            drafter.reset()
            self._drafter_supports_partial_rm = bool(result)
        except Exception:
            self._drafter_supports_partial_rm = False
        return self._drafter_supports_partial_rm

    def _undo_drafter_tokens(self, target_n: int):
        """Undo drafter tokens back to target_n using kv_cache_seq_rm or reset."""
        if self.drafter_llm is None:
            return
        drafter = self.drafter_llm
        current_n = drafter.n_tokens
        if current_n <= target_n:
            return
        if self._test_drafter_partial_rm():
            drafter._ctx.kv_cache_seq_rm(0, target_n, current_n)
            drafter.n_tokens = target_n
        else:
            saved_ids = list(self.llm.input_ids[:target_n])
            drafter.reset()
            drafter.eval(saved_ids)

    def _draft_tokens(self, num_pred: int) -> List[int]:
        """Generate K draft tokens from the drafter model."""
        if self.drafter_llm is None:
            return []
        drafter = self.drafter_llm
        draft_n = drafter.n_tokens
        draft_tokens = []
        for _ in range(num_pred):
            tok = drafter.sample(temp=0.0)
            draft_tokens.append(tok)
            drafter.eval([tok])
        return draft_tokens

    def tokenize(self, text: str, add_bos: bool = False) -> List[int]:
        if isinstance(text, str):
            text = text.encode()
        return list(self.llm.tokenize(text, add_bos=add_bos))

    def detokenize(self, token_ids: List[int]) -> str:
        return self.llm.detokenize(token_ids).decode("utf-8", errors="replace")

    def reset(self):
        self.llm.reset()
        if self.drafter_llm is not None:
            self.drafter_llm.reset()

    def eval(self, token_ids: List[int]):
        self.llm.eval(token_ids)
        if self.drafter_llm is not None:
            self.drafter_llm.eval(token_ids)

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
        num_pred: int = 3,
        rep_threshold: int = 3,
        min_tokens_before_stop: int = 5,
    ) -> Tuple[str, List[int], Dict]:
        """Generate tokens using speculative decoding.

        Must call eval(context_tokens) before calling this method.
        Both target and drafter caches must be in sync.

        Returns: (text, gen_ids, stats)
        """
        gen_ids: List[int] = []
        stop_strs = [stop_str, "Assistant:", "assistant:"]
        stats = {
            "draft_proposed": 0,
            "draft_accepted": 0,
            "total_tokens": 0,
            "steps": 0,
            "rejections": 0,
        }

        while len(gen_ids) < max_new:
            stats["steps"] += 1

            # Step 1: Get drafter predictions BEFORE sampling target
            draft_tokens = self._draft_tokens(num_pred)
            drafter_n_after_draft = self.drafter_llm.n_tokens if self.drafter_llm else 0

            # Step 2: Sample target token
            target_token = self.sample()

            if not draft_tokens or draft_tokens[0] != target_token:
                # No draft match: just eval the target token
                gen_ids.append(target_token)
                self.llm.eval([target_token])
                # Sync drafter: undo draft tokens, eval accepted token
                if self.drafter_llm is not None:
                    self._undo_drafter_tokens(drafter_n_after_draft - num_pred)
                    self.drafter_llm.eval([target_token])
                if draft_tokens:
                    stats["draft_proposed"] += len(draft_tokens)
            else:
                # Draft[0] matches target!
                stats["draft_proposed"] += len(draft_tokens)
                stats["draft_accepted"] += 1
                accepted = [target_token]
                self.llm.eval([target_token])

                # Verify remaining draft tokens
                for i in range(1, len(draft_tokens)):
                    next_target = self.sample()
                    if next_target == draft_tokens[i]:
                        accepted.append(draft_tokens[i])
                        self.llm.eval([draft_tokens[i]])
                        stats["draft_accepted"] += 1
                    else:
                        stats["rejections"] += 1
                        accepted.append(next_target)
                        self.llm.eval([next_target])
                        break

                gen_ids.extend(accepted)
                # Sync drafter: undo draft tokens, eval accepted tokens
                if self.drafter_llm is not None:
                    self._undo_drafter_tokens(drafter_n_after_draft - num_pred)
                    self.drafter_llm.eval(accepted)

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

    def generate_simple(self, max_new: int = 60, stop_str: str = "\nUser:",
                        rep_threshold: int = 3, min_tokens_before_stop: int = 5) -> Tuple[str, List[int]]:
        """Simple generation without speculation (for baseline)."""
        gen_ids: List[int] = []
        stop_strs = [stop_str, "Assistant:", "assistant:"]

        while len(gen_ids) < max_new:
            tok = self.sample()
            gen_ids.append(tok)
            self.llm.eval([tok])

            cur = self.detokenize(gen_ids)
            if len(gen_ids) >= min_tokens_before_stop:
                for s in stop_strs:
                    if s in cur:
                        return cur.split(s)[0], gen_ids
            if self._detect_repetition(cur, threshold=rep_threshold):
                break

        return self.detokenize(gen_ids), gen_ids
