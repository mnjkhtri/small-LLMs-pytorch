import torch
import torch.nn as nn
from contextlib import contextmanager, nullcontext
from typing import List
from dataclasses import dataclass


@dataclass
class GenerationStep:
    token_id: int
    top_k_ids: List[int]
    top_k_probs: List[float]

class ActivationCache:
    def __init__(self, cpu: bool = True, mode: str = "last"):
        assert mode in ("last", "list")
        self.cpu = cpu; self.mode = mode; self.data = {}
    def save(self, name: str, x: torch.Tensor) -> None:
        y = x.detach()
        if self.cpu: y = y.cpu()
        if self.mode == "last": self.data[name] = y
        else: self.data.setdefault(name, []).append(y)

class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = ""
        self._get_cache = None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._get_cache is not None:
            cache = self._get_cache()
            if cache is not None: cache.save(self.name, x)
        return x

class InferenceMixin:
    def _canonicalize_hook_name(self, module_path: str) -> str:
        return module_path.replace("hook", "").replace("blocks.", "L")

    def setup_hookpoints(self) -> None:
        self._active_cache = None
        for module_path, m in self.named_modules():
            if isinstance(m, HookPoint):
                m.name = self._canonicalize_hook_name(module_path)
                m._get_cache = (lambda self=self: self._active_cache)

    @contextmanager
    def _caching(self, cache: ActivationCache):
        self._active_cache = cache
        try: yield
        finally: self._active_cache = None

    def _stream_tokens(
        self,
        ids,
        *,
        max_new_tokens: int,
        top_k: int,
        T: float,
        eos_token_id=None,
        cache: ActivationCache | None = None,
    ):
        device = next(self.parameters()).device
        past = None
        ctx = nullcontext() if cache is None else self._caching(cache)

        with torch.no_grad(), ctx:
            inp = torch.tensor([ids], device=device, dtype=torch.long)
            out = self.forward(inp, past_key_values=None, use_cache=True)
            logits, past = out["logits"], out["past_key_values"]

            for _ in range(max_new_tokens):
                next_logits = logits[:, -1, :]
                k = min(top_k, next_logits.shape[-1])

                probs_raw = torch.softmax(next_logits, dim=-1)
                vis_probs, vis_ids = torch.topk(probs_raw, k)

                if T <= 0:
                    next_id = int(torch.argmax(next_logits, dim=-1).item())
                else:
                    topk_logits, _ = torch.topk(next_logits, k)
                    cutoff = topk_logits[..., -1].unsqueeze(-1)
                    masked = torch.where(next_logits < cutoff, torch.full_like(next_logits, -float("inf")), next_logits)
                    probs_sample = torch.softmax(masked / T, dim=-1)
                    next_id = int(torch.multinomial(probs_sample, 1)[0].item())

                yield GenerationStep(next_id, vis_ids[0].tolist(), vis_probs[0].tolist())

                if eos_token_id is not None and next_id == eos_token_id:
                    break

                inp = torch.tensor([[next_id]], device=device, dtype=torch.long)
                out = self.forward(inp, past_key_values=past, use_cache=True)
                logits, past = out["logits"], out["past_key_values"]

    def stream_generate(
        self,
        prompt_ids,
        *,
        max_new_tokens: int = 256,
        top_k: int = 10,
        T: float = 0.8,
        eos_token_id=None,
    ):
        ids = prompt_ids.copy()
        for step in self._stream_tokens(ids, max_new_tokens=max_new_tokens, top_k=top_k, T=T, eos_token_id=eos_token_id, cache=None):
            ids.append(step.token_id)
            yield step

    def stream_generate_with_cache(
        self,
        prompt_ids,
        *,
        max_new_tokens: int = 256,
        top_k: int = 10,
        T: float = 0.8,
        eos_token_id=None,
        cache_mode: str = "last",
    ):
        cache = ActivationCache(cpu=True, mode=cache_mode)
        ids = prompt_ids.copy()
        for step in self._stream_tokens(ids, max_new_tokens=max_new_tokens, top_k=top_k, T=T, eos_token_id=eos_token_id, cache=cache):
            ids.append(step.token_id)
            yield step, cache
