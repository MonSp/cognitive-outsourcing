"""SpeculativeDecodingCompiler — MeaningCompiler with drafter model support.

Extends MeaningCompiler to support llama.cpp's native speculative decoding
via LlamaDraftModel. Enables SIG + SpecDec compound acceleration experiments.

Key design: SpecDec only works through Llama.generate() which handles the
draft-verify cycle internally. The MeaningCompiler's sample()+eval() loop
bypasses this, so we override generate_until_str() to use Llama.generate()
when a drafter is present.

Supported drafter types:
- Prompt lookup (n-gram): Fast, no extra VRAM, works on all hardware
- Model-based: Uses a smaller model, higher acceptance rate, extra VRAM
"""

import time
import numpy as np
import numpy.typing as npt
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from llama_cpp import Llama, LlamaDraftModel
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

from .compiler import MeaningCompiler, SEQ_ID


@dataclass
class GenerationStepRecord:
    """Records acceptance information for a single generation step."""
    step_idx: int
    tokens_before: int
    tokens_after: int
    tokens_accepted: int
    draft_tokens_accepted: int
    wall_time_s: float


class ModelDraftModel(LlamaDraftModel):
    """Concrete LlamaDraftModel using a smaller Llama model as drafter.

    Note: This requires significant extra VRAM. Use LlamaPromptLookupDecoding
    for a zero-overhead alternative.
    """

    def __init__(
        self,
        drafter_path: str,
        n_ctx: int = 16384,
        n_gpu_layers: int = 99,
        n_threads: int = 4,
        draft_n_predict: int = 3,
    ):
        self.drafter = Llama(
            model_path=drafter_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False,
        )
        self.draft_n_predict = draft_n_predict

    def __call__(
        self, input_ids: npt.NDArray[np.intc], /, **kwargs
    ) -> npt.NDArray[np.intc]:
        """Generate draft tokens using the smaller model."""
        self.drafter._ctx.kv_cache_clear()
        self.drafter.n_tokens = 0
        self.drafter.eval(input_ids.tolist())

        draft_ids = []
        for _ in range(self.draft_n_predict):
            token = self.drafter.sample(temp=0.0)
            draft_ids.append(token)

        return np.array(draft_ids, dtype=np.intc)


class _NgramDraftModel(LlamaDraftModel):
    """Simple n-gram drafter that finds matching sequences in context.

    More robust than LlamaPromptLookupDecoding which has issues with
    Qwen3.5 models in llama-cpp-python.
    """

    def __init__(self, ngram_size: int = 3, num_pred: int = 3):
        self.ngram_size = ngram_size
        self.num_pred = num_pred

    def __call__(self, input_ids: npt.NDArray[np.intc], /, **kwargs) -> npt.NDArray[np.intc]:
        if len(input_ids) < self.ngram_size + 1:
            return np.array([], dtype=np.intc)

        ngram = input_ids[-self.ngram_size:]
        for i in range(len(input_ids) - self.ngram_size - 1, -1, -1):
            if np.array_equal(input_ids[i:i + self.ngram_size], ngram):
                start = i + self.ngram_size
                end = min(start + self.num_pred, len(input_ids))
                if start < end:
                    return np.array(input_ids[start:end], dtype=np.intc).copy()
        return np.array([], dtype=np.intc)


class _EmptyDraftModel(LlamaDraftModel):
    """Empty drafter that returns no draft tokens.

    Goes through the generate() path without actual speculative decoding.
    Used to measure the generate()-path overhead vs sample()+eval() path.
    """

    def __call__(self, input_ids: npt.NDArray[np.intc], /, **kwargs) -> npt.NDArray[np.intc]:
        return np.array([], dtype=np.intc)


def create_drafter(
    drafter_type: str = "prompt_lookup",
    drafter_path: Optional[str] = None,
    n_ctx: int = 16384,
    n_gpu_layers: int = 99,
    n_threads: int = 4,
    draft_n_predict: int = 3,
) -> Optional[LlamaDraftModel]:
    """Factory for creating drafter models.

    Args:
        drafter_type: "prompt_lookup" for n-gram, "model" for separate model,
                      "empty" for generate()-path baseline (no draft tokens)
        drafter_path: Path to drafter GGUF (required for "model" type)
        n_ctx, n_gpu_layers, n_threads: Model loading parameters
        draft_n_predict: Number of draft tokens to propose

    Returns:
        LlamaDraftModel instance or None
    """
    if drafter_type == "ngram":
        return _NgramDraftModel(ngram_size=3, num_pred=draft_n_predict)
    elif drafter_type == "prompt_lookup":
        return LlamaPromptLookupDecoding(
            max_ngram_size=3,
            num_pred_tokens=draft_n_predict,
        )
    elif drafter_type == "empty":
        return _EmptyDraftModel()
    elif drafter_type == "model":
        if not drafter_path:
            raise ValueError("drafter_path required for model-based drafter")
        return ModelDraftModel(
            drafter_path=drafter_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            draft_n_predict=draft_n_predict,
        )
    else:
        raise ValueError(f"Unknown drafter_type: {drafter_type}")


class SpeculativeDecodingCompiler(MeaningCompiler):
    """MeaningCompiler with optional drafter model for speculative decoding.

    When drafter is configured, overrides generate_until_str() to use
    Llama.generate() which handles the draft-verify cycle internally.
    """

    def __init__(
        self,
        model_path: str,
        drafter_type: str = "prompt_lookup",
        drafter_path: Optional[str] = None,
        draft_n_predict: int = 3,
        n_ctx: int = 16384,
        n_gpu_layers: int = 99,
        n_threads: int = 4,
        type_k: Optional[int] = None,
        type_v: Optional[int] = None,
    ):
        self.drafter_type = drafter_type
        self.drafter_path = drafter_path
        self.draft_n_predict = draft_n_predict

        # Build draft model
        draft_model = create_drafter(
            drafter_type=drafter_type,
            drafter_path=drafter_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            draft_n_predict=draft_n_predict,
        )

        drafter_desc = drafter_type if drafter_type in ("prompt_lookup", "empty", "ngram") else drafter_path
        print(
            f"Loading model {model_path} (ctx={n_ctx}, gpu_layers={n_gpu_layers}, "
            f"draft={drafter_desc}, draft_k={draft_n_predict})"
        )
        kwargs = dict(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            draft_model=draft_model,
            logits_all=True,  # Required for generate() path
        )
        if type_k is not None:
            kwargs["type_k"] = type_k
        if type_v is not None:
            kwargs["type_v"] = type_v

        self.llm = Llama(**kwargs)
        self.n_ctx = n_ctx
        self.type_k = type_k
        self.type_v = type_v

    @property
    def has_drafter(self) -> bool:
        return self.llm.draft_model is not None

    def generate_until_str(
        self,
        stop_str: str,
        max_new: int = 300,
        rep_threshold: int = 3,
        min_tokens_before_stop: int = 5,
    ) -> Tuple[str, List[int]]:
        """Generate tokens until stop condition, using SpecDec if drafter present."""
        if not self.has_drafter:
            return super().generate_until_str(
                stop_str, max_new, rep_threshold, min_tokens_before_stop)

        gen_ids: List[int] = []
        stop_strs = [stop_str, "Assistant:", "assistant:"]

        # generate() needs at least one token to continue from.
        # Pass the last token from the current KV cache context.
        # This token will be re-eval'd but that's acceptable.
        last_token = [self.llm._input_ids[self.n_tokens - 1]] if self.n_tokens > 0 else []

        gen = self.llm.generate(
            last_token,
            top_k=40, top_p=0.95, temp=0.0,
            reset=False,
        )

        for token in gen:
            gen_ids.append(token)
            cur = self.detokenize(gen_ids)

            if len(gen_ids) >= min_tokens_before_stop:
                for s in stop_strs:
                    if s in cur:
                        return cur.split(s)[0], gen_ids

            if self._detect_repetition(cur, threshold=rep_threshold):
                break

            if len(gen_ids) >= max_new:
                break

        return self.detokenize(gen_ids), gen_ids


class InstrumentedSpecDecCompiler(SpeculativeDecodingCompiler):
    """SpeculativeDecodingCompiler with per-step acceptance tracking.

    Tracks n_tokens changes to infer how many draft tokens were accepted.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_records: List[GenerationStepRecord] = []
        self._step_idx = 0

    def reset_records(self):
        self.step_records = []
        self._step_idx = 0

    def generate_until_str_instrumented(
        self,
        stop_str: str,
        max_new: int = 300,
        rep_threshold: int = 3,
        min_tokens_before_stop: int = 5,
    ) -> Tuple[str, List[int], List[GenerationStepRecord]]:
        """Generate with per-step acceptance tracking."""
        self.reset_records()

        if not self.has_drafter:
            text, ids = super().generate_until_str(
                stop_str, max_new, rep_threshold, min_tokens_before_stop)
            self.step_records.append(GenerationStepRecord(
                step_idx=0,
                tokens_before=self.n_tokens - len(ids),
                tokens_after=self.n_tokens,
                tokens_accepted=len(ids),
                draft_tokens_accepted=0,
                wall_time_s=0.0,
            ))
            return text, ids, self.step_records

        gen_ids: List[int] = []
        stop_strs = [stop_str, "Assistant:", "assistant:"]

        last_token = [self.llm._input_ids[self.n_tokens - 1]] if self.n_tokens > 0 else []

        gen = self.llm.generate(
            last_token,
            top_k=40, top_p=0.95, temp=0.0,
            reset=False,
        )

        for token in gen:
            t0 = time.time()
            tokens_before_step = self.n_tokens
            gen_ids.append(token)

            cur = self.detokenize(gen_ids)

            tokens_after_step = self.n_tokens
            tokens_accepted = tokens_after_step - tokens_before_step

            self.step_records.append(GenerationStepRecord(
                step_idx=self._step_idx,
                tokens_before=tokens_before_step,
                tokens_after=tokens_after_step,
                tokens_accepted=max(1, tokens_accepted),
                draft_tokens_accepted=max(0, tokens_accepted - 1),
                wall_time_s=time.time() - t0,
            ))
            self._step_idx += 1

            if len(gen_ids) >= min_tokens_before_stop:
                for s in stop_strs:
                    if s in cur:
                        return cur.split(s)[0], gen_ids, self.step_records

            if self._detect_repetition(cur, threshold=rep_threshold):
                break

            if len(gen_ids) >= max_new:
                break

        return self.detokenize(gen_ids), gen_ids, self.step_records

    def get_acceptance_summary(self) -> Dict[str, Any]:
        if not self.step_records:
            return {}

        total_accepted = sum(r.tokens_accepted for r in self.step_records)
        total_draft_accepted = sum(r.draft_tokens_accepted for r in self.step_records)
        total_steps = len(self.step_records)

        acceptance_rates = [
            r.tokens_accepted / max(1, r.tokens_accepted + r.draft_tokens_accepted)
            for r in self.step_records
            if r.tokens_accepted > 0
        ]

        return {
            "total_steps": total_steps,
            "total_tokens_accepted": total_accepted,
            "total_draft_tokens_accepted": total_draft_accepted,
            "avg_tokens_per_step": total_accepted / max(1, total_steps),
            "avg_acceptance_rate": (
                np.mean(acceptance_rates) if acceptance_rates else 0.0
            ),
            "total_wall_time_s": sum(r.wall_time_s for r in self.step_records),
        }
