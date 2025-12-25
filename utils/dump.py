import time
from contextlib import contextmanager


class Logger:
    def __init__(self, *, verbose=True, sink=print):
        self.verbose = verbose
        self.sink = sink

    def _p(self, s):
        if self.verbose: self.sink(s)

    def block(self, title): self._p(f"\n🧱 {title}")
    def step(self, msg): self._p(f"🔹 {msg}")

    def done(self, msg="Done.", elapsed=None, gap=0):
        if not self.verbose: return
        tail = f" ⏱️ {elapsed:.2f}s" if elapsed is not None else ""
        self.sink(f"✅ {msg}{tail}")
        if gap: self.sink("\n"*gap, end="")

    @contextmanager
    def timer(self):
        t0 = time.time()
        class T: pass
        t = T()
        try:
            yield t
        finally:
            t.dt = time.time() - t0

    def cuda_mem(self, tag=""):
        try:
            import torch
            if not torch.cuda.is_available(): return
            a = torch.cuda.memory_allocated() / 1e9
            r = torch.cuda.memory_reserved() / 1e9
            self.step(f"🧠 {tag}cuda mem a={a:.3f}GB r={r:.3f}GB")
        except Exception:
            return

LOGGER = Logger()
