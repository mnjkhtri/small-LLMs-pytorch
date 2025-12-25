import torch
import torch.nn as nn
import torch.nn.functional as F

from torchao.dtypes import to_affine_quantized_intx
from torchao.quantization.quant_primitives import MappingType


"""
Cookbook for a quantization scheme:
    - forward(x): run the layer using the quantized version
    - is_ok(m, path): return True if module m at path is eligible to be replaced by this class. [The from_pretrained asks for this]
    - build(m, torch_dtype): construct an empty replacement module. [from_pretrained]
    - pack(fp_tensor, ...): stateless FP to packed tensors conversion used by load_wts_
    - load_wts_(name, src): accept an FP named name, pack+copy into buffers; return True if handled. [Streamer]
"""


class Int8Linear(nn.Module):
    # Naive weight-only int8: stores (int8 W, per-row scale) and explicitly dequantizes to fp each forward.

    def __init__(self, in_features, out_features, scale_dtype):
        super().__init__()
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))   # quantized weights (row-major)
        self.register_buffer("scale",  torch.empty((out_features,), dtype=scale_dtype))             # per-output-channel scale

    def forward(self, x: torch.Tensor):
        # Dequantize full W every call (slow + alloc-heavy); correctness baseline, not a fast kernel path.
        w = self.weight.to(x.dtype) * self.scale.to(x.dtype).unsqueeze(1)
        return F.linear(x, w)  # uses normal fp GEMM since w is fp

    @classmethod
    def is_ok(cls, m, path):
        return isinstance(m, nn.Linear) and m.bias is None  # bias omitted to keep this minimal

    @classmethod
    def build(cls, m, torch_dtype):
        if m.bias is not None:
            raise ValueError("Int8Linear supports bias=False only")
        return cls(m.in_features, m.out_features, scale_dtype=torch_dtype)  # keep scales in model compute dtype

    @staticmethod
    def pack(w_fp: torch.Tensor, qmax=127, eps=1e-8):
        # Symmetric per-row affine quantization: w ≈ scale[row] * q[row], with q in [-127, 127].
        w32 = w_fp.float()
        amax = w32.abs().amax(dim=1)                        # per-row (out_features) max magnitude
        scale = (amax/qmax).clamp(min=eps)                  # avoid div-by-zero; eps just stabilizes
        q = torch.round(w32 / scale.unsqueeze(1)).clamp(-qmax, qmax).to(torch.int8)  # quantize each row
        z = (amax==0)                                       # exact-zero rows
        if z.any(): q[z] = 0; scale[z] = 1.0                # any scale works if all zeros; pick 1.0
        return q, scale

    def load_wts_(self, name: str, src: torch.Tensor):
        # Streamer hook: accepts fp 'weight', packs to (int8, scale), copies into buffers.
        if name != "weight":
            return False
        if tuple(src.shape) != tuple(self.weight.shape):
            raise ValueError(f"shape mismatch fp weight -> int8: {tuple(src.shape)} vs {tuple(self.weight.shape)}")
        q, sc = self.pack(src)
        self.weight.copy_(q.to(device=self.weight.device), non_blocking=True)  # int8 payload
        self.scale.copy_(sc.to(dtype=self.scale.dtype, device=self.scale.device), non_blocking=True)  # scales
        return True


class TorchAOInt8Linear(nn.Module):
    # TorchAO weight-only int8: stores an AffineQuantizedTensor so kernels/compile can fuse dequantization.

    def __init__(self, in_features, out_features, *, bias: bool, scale_dtype, group_size=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size                             # None => per-row; else per-group along K
        self.scale_dtype = scale_dtype
        # Placeholder param on 'meta' so you can instantiate on meta and later materialize without allocating fp weights.
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=torch.int8, device="meta"),
            requires_grad=False
        )
        self.bias = None
        if bias:
            # Bias stays fp; only weights are quantized.
            self.bias = nn.Parameter(torch.empty((out_features,), dtype=scale_dtype, device="meta"), requires_grad=False)

    def forward(self, x: torch.Tensor):
        # If self.weight is an AffineQuantizedTensor, dispatch/compile can generate fused int8-weight kernels.
        return F.linear(x, self.weight, self.bias)

    @classmethod
    def is_ok(cls, m, path):
        return isinstance(m, nn.Linear)  # supports bias and bias-free

    @classmethod
    def build(cls, m, torch_dtype, *, group_size=None):
        # Builds an empty shell; real quantization happens when weights arrive in load_wts_.
        return cls(m.in_features, m.out_features, bias=(m.bias is not None), scale_dtype=torch_dtype, group_size=group_size)

    @staticmethod
    def pack(w_fp: torch.Tensor, *, group_size=None, scale_dtype=None):
        # Creates an AffineQuantizedTensor (int8 payload + scales/zero-points) using symmetric mapping.
        if group_size is None:
            block_size = (1, w_fp.shape[1])                      # per-row (per-output-channel) quantization
        else:
            block_size = (1, int(group_size))                    # per-group quantization along K dimension
        return to_affine_quantized_intx(
            w_fp,
            MappingType.SYMMETRIC,                               # symmetric affine (typically zero_point ~ 0)
            block_size,
            torch.int8,
            scale_dtype=scale_dtype,                             # keep scales in bf16/fp16 to match model dtype
            zero_point_dtype=torch.int64,                        # stored zps (even if symmetric) in an integer dtype
        )

    def load_wts_(self, name: str, src: torch.Tensor):
        # Streamer hook: quantize as weights are streamed, on the module's current device (CPU or CUDA).
        if name == "weight":
            dev = self.weight.device
            w = src.to(device=dev, non_blocking=True)            # move fp shard to where you want to quantize
            q = self.pack(w, group_size=self.group_size, scale_dtype=self.scale_dtype)
            # Replace placeholder Parameter with the quantized tensor (still wrapped as Parameter for state_dict plumbing).
            self.weight = nn.Parameter(q, requires_grad=False)
            return True

        if name == "bias":
            # Bias is loaded as fp; quant path doesn't touch it.
            if self.bias is None:
                raise ValueError("dst has no bias but got bias")
            self.bias.data.copy_(src.to(dtype=self.bias.dtype, device=self.bias.device, non_blocking=True))
            return True

        return False
    

QUANT_MAP = {
    "int8_naive": Int8Linear,       # dequantizes W every forward; simplest correctness baseline
    "int8": TorchAOInt8Linear       # keeps quantized weight tensor; relies on dispatch for speed
}
