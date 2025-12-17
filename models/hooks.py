import torch
import torch.nn as nn
from contextlib import contextmanager


class Cache:
    """A tiny activation cache.

    Args:
        cpu: If True, store cached tensors on CPU to save GPU memory.
        mode: "last" keeps only the most recent tensor per name.
              "list" appends all saved tensors per name.
    """
    def __init__(self, cpu: bool = True, mode: str = "last"):
        assert mode in ("last", "list")
        self.cpu = cpu
        self.mode = mode
        self.data = {}

    def save(self, name: str, x: torch.Tensor) -> None:
        # Detach so the cache doesn't keep autograd graphs alive.
        y = x.detach()
        if self.cpu:
            y = y.cpu()

        if self.mode == "last":
            self.data[name] = y
        else:
            self.data.setdefault(name, []).append(y)


class HookPoint(nn.Module):
    """A passthrough module that can save activations into an active Cache."""
    def __init__(self):
        super().__init__()
        self.name = ""
        self._get_cache = None  # a callable returning the current active cache (or None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If a cache context is active, save the activation under this hook's name.
        if self._get_cache is not None:
            cache = self._get_cache()
            if cache is not None:
                cache.save(self.name, x)
        return x


class HookedModelMixin:
    """Mixin that wires HookPoint modules into a cache context and provides helpers."""

    def _canonicalize_hook_name(self, module_path: str) -> str:
        # Optional normalization: strip "hook" from module names for nicer keys.
        n = module_path
        n = n.replace("hook", "")
        return n

    def setup_hookpoints(self) -> None:
        # Must be called after modules are constructed, so HookPoints can be named.
        self._active_cache = None
        for module_path, m in self.named_modules():
            if isinstance(m, HookPoint):
                m.name = self._canonicalize_hook_name(module_path)
                # Each hook reads "the current active cache" from the model at runtime.
                m._get_cache = (lambda self=self: self._active_cache)

    @contextmanager
    def _caching(self, cache: Cache):
        # Activate a cache for the duration of the context.
        self._active_cache = cache
        try:
            yield
        finally:
            self._active_cache = None


    def _stream_tokens(
        self,
        ids,
        cache,
        *,
        max_new_tokens: int,
        top_k: int,
        T: float,
        eos_token_id=None,
    ):
        device = next(self.parameters()).device
        past = None

        with torch.no_grad(), self._caching(cache):

            # Prime the model on the full prompt once.
            inp = torch.tensor([ids], device=device, dtype=torch.long)
            out = self.forward(inp, past_key_values=None, use_cache=True)
            logits, past = out["logits"], out["past_key_values"]

            for _ in range(max_new_tokens):

                # Next-token distribution from the last position.
                next_logits = logits[:, -1, :]

                # Top-k filtering: keep only the k largest logits.
                k = min(top_k, next_logits.shape[-1])
                topk_logits, _ = torch.topk(next_logits, k)
                cutoff = topk_logits[..., -1].unsqueeze(-1)
                masked = torch.where(
                    next_logits < cutoff,
                    torch.full_like(next_logits, -float("inf")),
                    next_logits,
                )

                # Sampling:
                probs = torch.softmax(masked / T, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1)[0].item())

                ids.append(next_id)

                yield next_id

                # Early stop if we hit EOS.
                if eos_token_id is not None and next_id == eos_token_id: break

                inp = torch.tensor([[next_id]], device=device, dtype=torch.long)
                out = self.forward(inp, past_key_values=past, use_cache=True)
                logits, past = out["logits"], out["past_key_values"]


    def generate_with_cache(
        self,
        prompt_ids,
        *,
        max_new_tokens: int = 256,
        top_k: int = 100,
        T: float = 0.8,
        eos_token_id=None,
        cache_mode: str = "list",
    ):

        cache = Cache(cpu=True, mode=cache_mode)
        ids = prompt_ids

        for _ in self._stream_tokens(
            ids,
            cache,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            T=T,
            eos_token_id=eos_token_id,
        ):
            pass

        return ids, cache


    def stream_generate_with_cache(
        self,
        prompt_ids,
        *,
        max_new_tokens: int = 256,
        top_k: int = 100,
        T: float = 0.8,
        eos_token_id=None,
        cache_mode: str = "last",
    ):

        cache = Cache(cpu=True, mode=cache_mode)
        ids = prompt_ids

        for next_id in self._stream_tokens(
            ids,
            cache,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            T=T,
            eos_token_id=eos_token_id,
        ):
            yield next_id, cache


    def infer_with_cache(self, prompt_ids):
        # Single forward pass with caching enabled.
        cache = Cache(cpu=True, mode="last")
        device = next(self.parameters()).device
        input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)

        with self._caching(cache):
            out = self.forward(input_ids, past_key_values=None, use_cache=False)

        return out, cache
    