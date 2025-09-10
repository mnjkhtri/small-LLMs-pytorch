import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from pathlib import Path
from tqdm import tqdm
from safetensors import safe_open
import gc
from huggingface_hub import hf_hub_download
import os
import math


import torch
import math
from typing import Optional

# self.head_dim, self.rope_theta, self.max_length, freq_config=self.rope_scaling, device=qx.device, dtype=qx.dtype)

def llama3_rope_cos_sin(
    head_dim,
    rope_theta,
    max_length,
    rope_scaling,
    position_ids: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = torch.float32,
):
    """
    Computes the RoPE cos and sin embeddings for LLaMA 3.1 in a single function.

    Args:
        config: PretrainedConfig with rope settings.
        position_ids (torch.Tensor): Position indices (batch, seq).
        device (torch.device, optional): Torch device.
        dtype (torch.dtype, optional): Output dtype.

    Returns:
        (cos, sin): Tensors of shape (batch, seq, dim).
    """
    device = device or position_ids.device
    base = rope_theta
    partial_rotary_factor = 1.0
    dim = int(head_dim * partial_rotary_factor)

    # Inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))

    # RoPE scaling params
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    # Expand for batch/seq
    inv_freq_expanded = inv_freq_llama[None, :, None].expand(position_ids.shape[0], -1, 1).to(device)
    position_ids_expanded = position_ids[:, None, :].float()

    # Compute cos/sin
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)

    return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Llama3Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.tok_embed = nn.Embedding(self.vocab_size, self.embed_dim)

    @property
    def weight(self): # convenience alias
        return self.tok_embed.weight
    
    def forward(self, x):
        # [B, SEQ_LENGTH]
        return self.tok_embed(x)
        # [B, SEQ_LENGTH, EMBED_DIM]

class Llama3LMHead(nn.Module):
    def __init__(self, embed_dim, embedding):
        super().__init__()
        self.norm = nn.RMSNorm(embed_dim, eps=1e-5)
        self.embedding = embedding

    def forward(self, x):
        # [B, SEQ_LENGTH, EMBED_DIM]
        w = self.embedding.weight
        x = self.norm(x)
        out = F.linear(x, w)
        return out
        # [B, SEQ_LENGTH, VOCAB_SIZE]

class MHAttention(nn.Module):

    def __init__(self, max_length, embed_dim, num_heads, num_kv_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.max_length = max_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = (embed_dim // num_heads)
        self.group_size = (num_heads // num_kv_heads)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, num_kv_heads * self.head_dim, bias=False)
        # num_heads q share num_kv_heads kv;

        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.rope_theta = 500_000.0

        self.rope_scaling = {
            "type": "llama3",              # aka "rope_type" in config
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_context_length": 8192,
            "original_max_position_embeddings": 8192,
        }

    def forward(self, x, past_kv=None, use_cache=False):
        B, T_q, _ = x.size()
        qx = self.q_proj(x)
        kx = self.k_proj(x)
        vx = self.v_proj(x)

        qx = qx.view(B, T_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)         # [B, H, T_q, Dh]
        kx = kx.view(B, T_q, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)      # [B, Hkv, T_q, Dh]
        vx = vx.view(B, T_q, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)      # [B, Hkv, T_q, Dh]

        # --- NEW: RoPE (no buffers) ---
        # figure out how many past tokens exist (for correct absolute positions)
        T_past = 0 if past_kv is None else past_kv[0].size(2)
        # we only need cos/sin for positions [T_past .. T_past+T_q-1]

        position_ids = torch.arange(T_past, T_past + T_q, device=qx.device)   # [T_q]
        position_ids = position_ids.unsqueeze(0).expand(B, -1)                # [B, T_q]

        cos, sin = llama3_rope_cos_sin(self.head_dim, self.rope_theta, self.max_length, rope_scaling=self.rope_scaling, position_ids=position_ids, device=qx.device, dtype=qx.dtype)

        # qx = compute_rope(qx, cos, sin)
        # kx = compute_rope(kx, cos, sin)

        qx, kx = apply_rotary_pos_emb(qx, kx, cos, sin)
        # --- END NEW ---

        # for compute expand them:
        kx = kx.repeat_interleave(self.group_size, dim=1)   # [B, H, T_q, Dh]
        vx = vx.repeat_interleave(self.group_size, dim=1)   # [B, H, T_q, Dh]

        if past_kv is not None:
            kp, vp = past_kv                                                # [B, H, T_past, Dh]
            T_past = kp.size(2)
            Kx = torch.cat([kp, kx], dim=2)                                 # [B, H, T_past + T_q, Dh]
            Vx = torch.cat([vp, vx], dim=2)                                 # [B, H, T_past + T_q, Dh]
            attn_mask = (torch.arange(T_past + T_q, device=qx.device).view(1, 1, 1, -1) <= (T_past + torch.arange(T_q, device=qx.device).view(1, 1, -1, 1)))
            # [1, 1, T_q, T_past + T_q]
        else:
            Kx, Vx = kx, vx
            T_past = 0
            attn_mask = (torch.arange(T_q, device=qx.device).view(1, 1, 1, -1) <= torch.arange(T_q, device=qx.device).view(1, 1, -1, 1))
            # [1, 1, T_q, T_q]

        # LOW:
        # att_w = torch.einsum('bhqd,bhkd->bhqk', [qx, Kx]) / (self.head_dim ** 0.5)  # [B, H, T_q, T_past + T_q]
        # att_w = att_w.masked_fill(~attn_mask, torch.finfo(att_w.dtype).min)         # [B, H, T_q, T_past + T_q]
        # att_w = F.softmax(att_w, dim=-1)                                            # [B, H, T_q, T_past + T_q]

        # HIGH
        att_w = torch.einsum('bhqd,bhkd->bhqk', [qx.float(), Kx.float()]) / (self.head_dim ** 0.5)
        att_w = att_w.masked_fill(~attn_mask, torch.finfo(att_w.dtype).min)
        att_w = F.softmax(att_w, dim=-1).to(qx.dtype)

        out = torch.einsum('bhal,bhlv->bhav', [att_w, Vx])  # [B, H, T_q, Dh]
        out = out.permute(0, 2, 1, 3).contiguous()          # [B, T_q, H, Dh]
        out = out.reshape(B, T_q, self.num_heads * self.head_dim)

        # [B, T_q, Edim]
        out = self.o_proj(out)
        # residue dropout

        if use_cache:
            return out, (Kx, Vx)
        
        return out
        
class MLPF(nn.Module):

    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim

        self.up_proj = nn.Linear(self.embed_dim, self.ff_dim, bias=False)     # "up"
        self.gate_proj = nn.Linear(self.embed_dim, self.ff_dim, bias=False)   # "gate"
        self.down_proj = nn.Linear(self.ff_dim, self.embed_dim, bias=False)   # "down"

    def forward(self, x):
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        x = self.down_proj(up * F.silu(gate))
        return x
        #  SwiGLU activation

class Llama3Block(nn.Module):
    def __init__(self, max_length, embed_dim, ff_dim, num_heads, num_kv_heads):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.mhattn = MHAttention(self.max_length, self.embed_dim, self.num_heads, self.num_kv_heads)
        self.mlpf = MLPF(self.embed_dim, self.ff_dim)

        self.norm1 = nn.RMSNorm(embed_dim, eps=1e-5)
        self.norm2 = nn.RMSNorm(embed_dim, eps=1e-5)

    def forward(self, x, past_kv=None, use_cache=False):
        if use_cache:
            attd_x, present_kv = self.mhattn(self.norm1(x), past_kv=past_kv, use_cache=True)
            x = x + attd_x
            x = x + self.mlpf(self.norm2(x))
            return x, present_kv
        else:
            x = x + self.mhattn(self.norm1(x))
            x = x + self.mlpf(self.norm2(x))
            return x
        
class Llama3(nn.Module):

    def __init__(self, vocab_size, max_length, embed_dim, ff_dim, num_heads, num_kv_heads, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers

        self.embedding = Llama3Embedding(self.vocab_size, self.embed_dim)
        self.lm_head = Llama3LMHead(self.embed_dim, self.embedding) # sending embedding to tie weights

        self.blocks = nn.ModuleList(
            [Llama3Block(max_length, embed_dim, ff_dim, num_heads, num_kv_heads) for _ in range(self.num_layers)]
        )

    def forward(self, x, past_key_values=None, use_cache=False):
        B, T = x.size()
        if past_key_values is not None and past_key_values[0] is not None:
            # K: [B, H, T_past, Dh]
            T_past = past_key_values[0][0].size(2)
        else:
            T_past = 0
        assert (T + T_past) <= self.max_length, "sequence length cant exceed max length"
        
        # [B, SEQ_LENGTH]
        x = self.embedding(x)
        # [B, SEQ_LENGTH, EMBED_DIM]

        if use_cache:
            if past_key_values is None:
                past_key_values = [None] * self.num_layers
            new_past_key_values = []
            for i, block in enumerate(self.blocks):
                x, present_kv = block(x, past_kv=past_key_values[i], use_cache=True)
                new_past_key_values.append(present_kv)

            x = self.lm_head(x)

            return {
                'logits': x,
                'past_key_values': new_past_key_values
            }

        else:
            for block in self.blocks:
                x = block(x)

            # [B, SEQ_LENGTH, EMBED_DIM]
            x = self.lm_head(x)
            # [B, SEQ_LENGTH, VOCAB_SIZE]

            return {
                'logits': x,
            }

    @classmethod
    def from_pretrained(cls, model_type='1b', *, torch_dtype=torch.bfloat16, device='cuda'):

        assert device == 'cuda' and torch_dtype == torch.bfloat16, "not really tested otherwise due to gpu poor"

        config = {
            "1b": dict(
                vocab_size=128256, max_length=8192, embed_dim=2048, ff_dim=4*2048, num_heads=32, num_kv_heads=8, num_layers=16
            )
        }[model_type]

        # Lets replace this with HF:
        MODEL_URL = "https://huggingface.co/meta-llama/llama-3.2-1b/resolve/main/model.safetensors"
        save_path = Path("weights_cache") / "llama-3.2-1b"
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / "model.safetensors"

        HF_TOKEN = os.getenv('HF_TOKEN')
        if not model_file.exists():
            # Stream download with progress bar
            headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
            with requests.get(MODEL_URL, headers=headers, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(model_file, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, desc="Downloading weights for Llama3"
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
                            pbar.update(len(chunk))

        with torch.device('meta'):
            model = cls(**config)

        all_mappings = {
            'model.embed_tokens.weight': 'embedding.tok_embed.weight',
            'model.norm.weight':         'lm_head.norm.weight'
        }

        blocks_map = lambda i: {
            f'model.layers.{i}.input_layernorm.weight':           f'blocks.{i}.norm1.weight',
            f'model.layers.{i}.mlp.up_proj.weight':               f'blocks.{i}.mlpf.up_proj.weight',
            f'model.layers.{i}.mlp.gate_proj.weight':             f'blocks.{i}.mlpf.gate_proj.weight',
            f'model.layers.{i}.mlp.down_proj.weight':             f'blocks.{i}.mlpf.down_proj.weight',
            f'model.layers.{i}.self_attn.q_proj.weight':          f'blocks.{i}.mhattn.q_proj.weight',
            f'model.layers.{i}.self_attn.k_proj.weight':          f'blocks.{i}.mhattn.k_proj.weight',
            f'model.layers.{i}.self_attn.v_proj.weight':          f'blocks.{i}.mhattn.v_proj.weight',
            f'model.layers.{i}.self_attn.o_proj.weight':          f'blocks.{i}.mhattn.o_proj.weight',
            f'model.layers.{i}.post_attention_layernorm.weight':  f'blocks.{i}.norm2.weight',
        }

        for i in range(config['num_layers']):
            all_mappings.update(blocks_map(i))

        # Stream tensors directly from the safetensors file into the model parameters (no big state dict)
        with torch.inference_mode():
            with safe_open(str(model_file), framework="pt", device="cpu") as f:
                f_keys = set(f.keys())
                step = 0
                for src_key, dst_path in all_mappings.items():
                    assert src_key in f_keys, f"Missing key in checkpoint: {src_key}"

                    t = f.get_tensor(src_key)  # CPU, memory-mapped

                    # Resolve destination parent and name
                    obj = model
                    parts = dst_path.split('.')
                    for p in parts[:-1]:
                        obj = getattr(obj, p)
                    name = parts[-1]

                    # Allocate REAL storage directly on target device
                    real = torch.empty_like(t, device=device, dtype=torch_dtype)
                    real.copy_(t, non_blocking=True)

                    # Install the tensor into the module (replaces meta param)
                    setattr(obj, name, torch.nn.Parameter(real, requires_grad=False))

                    # Drop CPU refs ASAP
                    del t

                    step += 1
                    if (step % 32) == 0:
                        gc.collect()

        gc.collect()
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()

        return model