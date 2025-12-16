import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager


class Cache:
    """
    A very small activation cache.

    What it stores:
      - key: a string name for a "hook site" (e.g., "blocks.1.attn.pattern")
      - value: a tensor captured at that site

    Two modes:
      - mode="last": cache[name] = most recent tensor (good for a single forward pass)
      - mode="list": cache[name] = [tensor_t0, tensor_t1, ...] (good for generation loops)

    Why cpu?
      - Attention patterns and residual streams can be large. Moving them to CPU avoids filling GPU memory
    """
    def __init__(self, cpu=True, mode="last"):
        assert mode in ("last", "list")
        self.cpu = cpu
        self.mode = mode
        self.data = {}

    def save(self, name, x):
        """
        Save tensor `x` under `name`.

        IMPORTANT:
        - This method does not modify x.
        - HookPoints will call this during forward passes.
        """
        y = x.detach()  # always detach
        if self.cpu:
            y = y.cpu()

        if self.mode == "last":
            self.data[name] = y
        else:
            self.data.setdefault(name, []).append(y)


class HookPoint(nn.Module):
    """
    HookPoint is an identity module that sits inside your model.

    It does NOT require user-registered hooks. Instead:
      - The root model temporarily "enables caching"
      - HookPoint asks the root model if caching is currently enabled
      - If yes, it writes its activation into the cache

    This makes usage extremely simple:
      out, cache = model.run_with_cache(x)
    without you manually attaching any hooks.

    How does it find the cache?
      - During model.setup_hookpoints(), each HookPoint gets a callback:
          self._get_cache() -> active Cache or None
      - If cache is None, HookPoint is a no-op (returns x).
    """
    def __init__(self):
        super().__init__()
        self.name = ""          # assigned automatically by root model
        self._get_cache = None  # function returning active cache or None

    def forward(self, x):
        """
        Called during model forward pass.

        Behavior:
          - If caching is enabled: save x into cache under self.name
          - Otherwise: do nothing
        """
        if self._get_cache is not None:
            cache = self._get_cache()
            if cache is not None:
                cache.save(self.name, x)
        return x


class HookedModelMixin:
    """
    Add this to your root model (GPT2).

    It provides:
      - setup_hookpoints(): auto-name HookPoints with standard transformer-ish names
      - run_with_cache(): run a forward pass and return (output, cache)
      - generate_with_cache(): generation loop that caches over time

    Caching policy:
      - If caching is enabled -> ALL HookPoints cache
      - If caching is disabled -> NONE cache
    """
    def _canonicalize_hook_name(self, module_path):
        n = module_path
        n = n.replace(".mhattn.", ".attn.")
        n = n.replace(".mlpf.", ".mlp.")
        n = n.replace("hook_logits", "logits")
        n = n.replace("embed.hook_embed", "embed")
        n = n.replace(".hook_resid_pre", ".resid_pre")
        n = n.replace(".hook_resid_mid", ".resid_mid")
        n = n.replace(".hook_resid_post", ".resid_post")
        n = n.replace(".attn.hook_attn", ".attn.pattern")
        n = n.replace(".mlp.hook_pre", ".mlp.pre")
        n = n.replace(".mlp.hook_post", ".mlp.post")
        n = n.replace(".mlp.hook_out", ".mlp.out")
        return n

    def setup_hookpoints(self):
        """
        Find all HookPoints, assign canonical names, and wire them to self._active_cache.
        """
        self._active_cache = None  # caching OFF by default

        for module_path, m in self.named_modules():
            if isinstance(m, HookPoint):
                m.name = self._canonicalize_hook_name(module_path)
                m._get_cache = (lambda self=self: self._active_cache)

    @contextmanager
    def _caching(self, cache):
        """
        Temporarily enable caching for this model.
        While active: ALL HookPoints will cache.
        """
        self._active_cache = cache
        try:
            yield
        finally:
            self._active_cache = None

    def run_with_cache(self, x):
        """
        Same signature as forward(), plus a cache argument.

        Usage:
          out, cache = model.run_with_cache(x)
          out, cache = model.run_with_cache(x, use_cache=True)  # if you want pkv returned too
        """
        cache = Cache(cpu=True, mode="last")

        with self._caching(cache):
            out = self.forward(x, past_key_values=None, use_cache=False)

        return out, cache
    