"""MeaningCompiler — lightweight local LLM wrapper around llama-cpp-python.

Merged from co_benchmark.py (base) and r2_benchmark.py (type_k/type_v,
n_tokens property, KV cache manipulation methods).
"""

from typing import List, Optional, Tuple
from llama_cpp import Llama

SEQ_ID = 0


class MeaningCompiler:
    """Wraps a llama.cpp model with tokenization, generation, and KV-cache
    management helpers used by the SIG injection pipeline.

    Class constants:
        TOOL_MARK: sentinel marking the start of a tool-call block.
        TOOL_END:  sentinel marking the end of a tool-call block.
    """

    TOOL_MARK = "<<<TOOL>>>"
    TOOL_END = "<<</TOOL>>>"

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        type_k: Optional[int] = None,
        type_v: Optional[int] = None,
    ):
        print(
            f"Loading model {model_path} (ctx={n_ctx}, gpu_layers={n_gpu_layers}, "
            f"type_k={type_k}, type_v={type_v})"
        )
        kwargs = dict(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        if type_k is not None:
            kwargs["type_k"] = type_k
        if type_v is not None:
            kwargs["type_v"] = type_v
        self.llm = Llama(**kwargs)
        self.n_ctx = n_ctx
        self.type_k = type_k
        self.type_v = type_v

    def tokenize(self, text: str, add_bos: bool = False) -> List[int]:
        return self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos)

    def detokenize(self, ids: List[int]) -> str:
        return self.llm.detokenize(ids).decode("utf-8", errors="replace")

    def reset_cache(self):
        self.llm._ctx.kv_cache_seq_rm(SEQ_ID, -1, -1)
        self.llm.n_tokens = 0

    def eval(self, tokens: List[int]):
        self.llm.eval(tokens)

    def rebuild_cache(self, token_ids: List[int]):
        self.reset_cache()
        self.eval(token_ids)

    def sample(self, temp: float = 0.0) -> int:
        token = self.llm.sample(temp=temp)
        self.eval([token])
        return token

    @property
    def n_tokens(self) -> int:
        return self.llm.n_tokens

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int) -> bool:
        return self.llm._ctx.kv_cache_seq_rm(seq_id, p0, p1)

    def kv_cache_seq_cp(self, src: int, dst: int, p0: int, p1: int):
        self.llm._ctx.kv_cache_seq_cp(src, dst, p0, p1)

    def kv_cache_seq_keep(self, seq_id: int):
        self.llm._ctx.kv_cache_seq_keep(seq_id)

    def set_n_tokens(self, n: int):
        """Set logical token position counter — required after kv_cache_seq_cp."""
        self.llm.n_tokens = n

    def clear_seq(self, seq_id: int):
        """Remove all KV cache entries for a specific sequence."""
        self.llm._ctx.kv_cache_seq_rm(seq_id, -1, -1)

    def _ids_endswith(self, seq: List[int], suffix: List[int]) -> bool:
        if len(suffix) > len(seq):
            return False
        return list(seq[-len(suffix):]) == list(suffix)

    def _detect_repetition(self, text: str, min_len: int = 6, threshold: int = 3) -> bool:
        if threshold < 1 or len(text) < min_len:
            return False
        for pat_len in range(min_len, min(40, len(text) // threshold)):
            tail = text[-pat_len:]
            if tail.strip() == "":
                continue
            if text.count(tail) >= threshold:
                return True
        lines = text.split("\n")
        if len(lines) >= 3:
            last_line = lines[-1].strip()
            if len(last_line) > 5:
                recent = [l.strip() for l in lines[-4:]]
                if recent.count(last_line) >= 2:
                    return True
        return False

    def generate_until_ids(
        self,
        stop_ids: List[int],
        max_new: int = 300,
        rep_threshold: int = 3,
    ) -> Tuple[str, List[int]]:
        gen_ids: List[int] = []
        for _ in range(max_new):
            token = self.sample()
            gen_ids.append(token)
            if self._ids_endswith(gen_ids, stop_ids):
                text_ids = gen_ids[: -len(stop_ids)]
                return self.detokenize(text_ids), gen_ids
            cur = self.detokenize(gen_ids)
            if "Assistant:" in cur or "assistant:" in cur:
                return cur, gen_ids
            if self._detect_repetition(cur, threshold=rep_threshold):
                break
        return self.detokenize(gen_ids), gen_ids

    def generate_until_str(
        self,
        stop_str: str,
        max_new: int = 300,
        rep_threshold: int = 3,
    ) -> Tuple[str, List[int]]:
        gen_ids: List[int] = []
        for _ in range(max_new):
            token = self.sample()
            gen_ids.append(token)
            cur = self.detokenize(gen_ids)
            if stop_str in cur:
                return cur.split(stop_str)[0], gen_ids
            if "Assistant:" in cur or "assistant:" in cur:
                return cur, gen_ids
            if self._detect_repetition(cur, threshold=rep_threshold):
                break
        return self.detokenize(gen_ids), gen_ids

    def generate_until_any(
        self,
        stop_strs: List[str],
        max_new: int = 300,
        rep_threshold: int = 3,
    ) -> Tuple[str, List[int], Optional[str]]:
        all_stops = list(stop_strs) + ["Assistant:", "assistant:"]
        gen_ids: List[int] = []
        for _ in range(max_new):
            token = self.sample()
            gen_ids.append(token)
            cur = self.detokenize(gen_ids)
            for s in all_stops:
                if s in cur:
                    return cur.split(s)[0], gen_ids, s
            if self._detect_repetition(cur, threshold=rep_threshold):
                break
        return self.detokenize(gen_ids), gen_ids, None

    def sanitize_generation(
        self,
        n_before: int,
        gen_text: str,
        gen_ids: List[int],
        cached_prefix_ids: List[int],
    ) -> Tuple[str, List[int], bool]:
        full_decoded = self.detokenize(gen_ids)
        need_rollback = False
        if "Assistant:" in full_decoded or "assistant:" in full_decoded.lower():
            need_rollback = True
        if not need_rollback and self._detect_repetition(full_decoded):
            need_rollback = True
        if not need_rollback:
            return gen_text, gen_ids, False
        self.rebuild_cache(cached_prefix_ids)
        return "", [], True


class PrefixCache:
    """KV-cache prefix reuse via sequence copying — AppLoop-PC baseline.

    Saves the KV cache of a shared prefix (e.g., system prompt + static tool
    descriptions) to a separate sequence, then restores it on subsequent turns
    to avoid re-prefilling the common prefix.  This approximates the behaviour
    of production prefix-caching systems (PagedAttention, RadixAttention)
    without requiring a full serving stack.

    Usage::

        pc = PrefixCache()
        # After initial prefill of the shared prefix ...
        pc.save(compiler)
        # Later, for each turn:
        pc.restore(compiler)
        compiler.eval(turn_specific_tokens)
        ...
    """

    PREFIX_SEQ = 1

    def __init__(self):
        self.prefix_len = 0
        self.saved_token_ids: List[int] = []

    def save(self, compiler: MeaningCompiler, token_ids: List[int]):
        """Persist the current KV cache as a reusable prefix."""
        self.prefix_len = compiler.n_tokens
        self.saved_token_ids = list(token_ids)
        compiler.clear_seq(self.PREFIX_SEQ)
        compiler.kv_cache_seq_cp(SEQ_ID, self.PREFIX_SEQ, 0, self.prefix_len)

    def restore(self, compiler: MeaningCompiler) -> List[int]:
        """Clear the current cache and restore the saved prefix.

        Returns the restored token IDs so callers can track context state.
        """
        compiler.reset_cache()
        compiler.kv_cache_seq_cp(self.PREFIX_SEQ, SEQ_ID, 0, self.prefix_len)
        compiler.set_n_tokens(self.prefix_len)
        return list(self.saved_token_ids)
