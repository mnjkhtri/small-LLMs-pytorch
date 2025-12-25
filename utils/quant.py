import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Cookbook for a quantization scheme:
    - forward(x): run the layer using the quantized version
    - is_ok(m, path): return True if module m at path is eligible to be replaced by this class. [The from_pretrained asks for this]
    - build(m, torch_dtype): construct an empty replacement module. [from_pretrained]
    - pack(fp_tensor, ...): stateless FP to packed tensors conversion used by load_wts_
    - load_wts_(name, src): accept an FP named name, pack+copy into buffers; return True if handled. [Streamer]
"""

class Int8Linear(nn.Module):

    def __init__(self, in_features, out_features, scale_dtype):
        super().__init__()
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale",  torch.empty((out_features,), dtype=scale_dtype))

    def forward(self, x: torch.Tensor):
        w = self.weight.to(x.dtype) * self.scale.to(x.dtype).unsqueeze(1)
        return F.linear(x, w)

    @classmethod
    def is_ok(cls, m, path):
        return isinstance(m, nn.Linear) and m.bias is None

    @classmethod
    def build(cls, m, torch_dtype):
        if m.bias is not None:
            raise ValueError("Int8Linear supports bias=False only")
        return cls(m.in_features, m.out_features, scale_dtype=torch_dtype)

    @staticmethod
    def pack(w_fp: torch.Tensor, qmax=127, eps=1e-8):
        w32 = w_fp.float()
        amax = w32.abs().amax(dim=1)
        scale = (amax/qmax).clamp(min=eps)
        q = torch.round(w32 / scale.unsqueeze(1)).clamp(-qmax, qmax).to(torch.int8)
        z = (amax==0)
        if z.any(): q[z] = 0; scale[z] = 1.0
        return q, scale

    def load_wts_(self, name: str, src: torch.Tensor):
        if name != "weight":
            return False
        if tuple(src.shape) != tuple(self.weight.shape):
            raise ValueError(f"shape mismatch fp weight -> int8: {tuple(src.shape)} vs {tuple(self.weight.shape)}")
        q, sc = self.pack(src)
        self.weight.copy_(q.to(device=self.weight.device), non_blocking=True)
        self.scale.copy_(sc.to(dtype=self.scale.dtype, device=self.scale.device), non_blocking=True)
        return True


QUANT_MAP = {
    "int8": Int8Linear
}