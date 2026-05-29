"""Qwen3.5-compatible generate() for llama-cpp-python.

Root cause: kv_cache_seq_rm(-1, X, -1) for non-zero X crashes on Qwen3.5.
Only full clear (X=0) works. This module provides a patched generate()
that uses reset() + re-eval instead of partial KV cache truncation.

Performance impact: re-eval of accepted prefix on rejection.
- 50 tokens: ~22ms
- 100 tokens: ~35ms
- 187 tokens: ~57ms
"""

import sys
import numpy as np
import numpy.typing as npt
from typing import Sequence, Optional, Generator


def patched_generate(
    llm,
    tokens: Sequence[int],
    top_k: int = 40,
    top_p: float = 0.95,
    min_p: float = 0.05,
    typical_p: float = 1.0,
    temp: float = 0.80,
    repeat_penalty: float = 1.0,
    reset: bool = True,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    tfs_z: float = 1.0,
    mirostat_mode: int = 0,
    mirostat_tau: float = 5.0,
    mirostat_eta: float = 0.1,
    penalize_nl: bool = True,
    logits_processor=None,
    stopping_criteria=None,
    grammar=None,
) -> Generator[int, Optional[Sequence[int]], None]:
    """Patched generate() that works with Qwen3.5.

    Key difference from original: uses reset() + re-eval for rejection
    instead of kv_cache_seq_rm partial truncation.
    """
    llm._mirostat_mu = __import__('ctypes').c_float(2.0 * mirostat_tau)
    llm._sampler = llm._init_sampler(
        top_k=top_k, top_p=top_p, min_p=min_p, typical_p=typical_p,
        temp=temp, repeat_penalty=repeat_penalty,
        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
        tfs_z=tfs_z, mirostat_mode=mirostat_mode,
        mirostat_tau=mirostat_tau, mirostat_eta=mirostat_eta,
        penalize_nl=penalize_nl, logits_processor=logits_processor,
        grammar=grammar,
    )

    if reset:
        llm.reset()

    sample_idx = llm.n_tokens + len(tokens) - 1
    tokens = list(tokens)

    while True:
        llm.eval(tokens)
        while sample_idx < llm.n_tokens:
            token = llm.sample(
                top_k=top_k, top_p=top_p, min_p=min_p, typical_p=typical_p,
                temp=temp, repeat_penalty=repeat_penalty,
                frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
                tfs_z=tfs_z, mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau, mirostat_eta=mirostat_eta,
                logits_processor=logits_processor, grammar=grammar,
                penalize_nl=penalize_nl, idx=sample_idx,
            )
            sample_idx += 1

            if stopping_criteria is not None and stopping_criteria(
                llm._input_ids[:sample_idx], llm._scores[sample_idx - llm.n_tokens, :]
            ):
                return

            tokens_or_none = yield token
            tokens.clear()
            tokens.append(token)
            if tokens_or_none is not None:
                tokens.extend(tokens_or_none)

            # REJECTION: use reset() + re-eval instead of kv_cache_seq_rm
            if sample_idx < llm.n_tokens and token != llm._input_ids[sample_idx]:
                accepted_prefix = list(llm.input_ids[:sample_idx])
                llm.reset()
                llm.eval(accepted_prefix)
                # sample_idx must match n_tokens after re-eval
                sample_idx = llm.n_tokens
                break

        if llm.draft_model is not None:
            llm.input_ids[llm.n_tokens : llm.n_tokens + len(tokens)] = tokens
            draft_tokens = llm.draft_model(
                llm.input_ids[: llm.n_tokens + len(tokens)]
            )
            tokens.extend(
                draft_tokens.astype(int)[: llm._n_ctx - llm.n_tokens - len(tokens)]
            )


class Qwen35CompatibleLlama:
    """Wrapper around Llama that provides Qwen3.5-compatible generate().

    Usage:
        llm = Qwen35CompatibleLlama("model.gguf", n_ctx=4096, n_gpu_layers=99,
                                     draft_model=my_drafter)
        for tok in llm.generate(tokens, temp=0.0):
            ...
    """

    def __init__(self, model_path, n_ctx=4096, n_gpu_layers=99, verbose=False,
                 draft_model=None, logits_all=None):
        from llama_cpp import Llama

        if logits_all is None:
            logits_all = draft_model is not None

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            draft_model=draft_model,
            logits_all=logits_all,
        )

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def __del__(self):
        del self._llm

    def generate(self, tokens, top_k=40, top_p=0.95, min_p=0.05,
                 typical_p=1.0, temp=0.80, repeat_penalty=1.0, reset=True,
                 frequency_penalty=0.0, presence_penalty=0.0, tfs_z=1.0,
                 mirostat_mode=0, mirostat_tau=5.0, mirostat_eta=0.1,
                 penalize_nl=True, logits_processor=None,
                 stopping_criteria=None, grammar=None):
        return patched_generate(
            self._llm, tokens,
            top_k=top_k, top_p=top_p, min_p=min_p, typical_p=typical_p,
            temp=temp, repeat_penalty=repeat_penalty, reset=reset,
            frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
            tfs_z=tfs_z, mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau, mirostat_eta=mirostat_eta,
            penalize_nl=penalize_nl, logits_processor=logits_processor,
            stopping_criteria=stopping_criteria, grammar=grammar,
        )

    def tokenize(self, text, add_bos=False):
        if isinstance(text, str):
            text = text.encode()
        return list(self._llm.tokenize(text, add_bos=add_bos))

    def detokenize(self, tokens):
        return self._llm.detokenize(tokens).decode("utf-8", errors="replace")

    def eval(self, tokens):
        self._llm.eval(tokens)

    def sample(self, temp=0.0):
        return self._llm.sample(temp=temp)

    def reset(self):
        self._llm.reset()

    @property
    def n_tokens(self):
        return self._llm.n_tokens

    @property
    def input_ids(self):
        return self._llm.input_ids

    @property
    def _llm_ref(self):
        return self._llm
