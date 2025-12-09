import sentencepiece as spm
from tokenizers import Tokenizer
from pathlib import Path
import requests
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class Gemma3Sentencepiece:
    def __init__(self, model_path):

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
    
    def _encode(self, text):
        return self.sp.encode(text, out_type=int, add_bos=True)

    def _encode_instruct(self, user: str, system: str = "You are a helpful assistant."):
        pass

    def decode(self, ids):
        # this ignores the special tokens by default
        return self.sp.decode(ids)

    @classmethod
    def gemma3(cls):
        TOKENIZER_URL = "https://huggingface.co/google/gemma-3-270m/resolve/main/tokenizer.model"
        save_path = Path(".cache") / "gemma-3-270m-instruct"
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / "tokenizer.model"

        HF_TOKEN = os.getenv('HF_TOKEN')
        if not model_file.exists():
            headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
            r = requests.get(TOKENIZER_URL, headers=headers, stream=True, timeout=60)
            r.raise_for_status()
            with open(model_file, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)

        return cls(str(model_file))

class Gemma3HFTokenizer:
    def __init__(self, model_path):

        self.tokenizer = Tokenizer.from_file(model_path)
    
    def _encode(self, text):
        return self.tokenizer.encode(text).ids

    def _encode_instruct(self, user: str, system: str = "You are a helpful assistant."):
        return self.tokenizer.encode(f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n").ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    @classmethod
    def gemma3(cls):
        TOKENIZER_URL = "https://huggingface.co/google/gemma-3-270m/resolve/main/tokenizer.json"
        save_path = Path(".cache") / "gemma-3-270m-instruct"
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / "tokenizer.json"

        HF_TOKEN = os.getenv('HF_TOKEN')
        if not model_file.exists():
            headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
            r = requests.get(TOKENIZER_URL, headers=headers, stream=True, timeout=60)
            r.raise_for_status()
            with open(model_file, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)

        return cls(str(model_file))
    
class Gemma3RMSNorm(torch.nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        dtype = x.dtype
        y = F.rms_norm(
            x.float(),
            (self.weight.numel(),),
            weight=(1.0 + self.weight.float()),
            eps=self.eps,
        )
        return y.to(dtype)

class Gemma3Embedding(nn.Module):
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
        return self.tok_embed(x) * (self.embed_dim ** 0.5)
        # [B, SEQ_LENGTH, EMBED_DIM]

class Gemma3LMHead(nn.Module):
    def __init__(self, embed_dim, embedding):
        super().__init__()
        self.norm = Gemma3RMSNorm(embed_dim, eps=1e-6)
        self.embedding = embedding

    def forward(self, x):
        # [B, SEQ_LENGTH, EMBED_DIM]
        w = self.embedding.weight
        x = self.norm(x)
        out = F.linear(x, w)
        return out
        # [B, SEQ_LENGTH, VOCAB_SIZE]

class Gemma3MLPF(nn.Module):

    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim

        self.up_proj = nn.Linear(self.embed_dim, self.ff_dim, bias=False)
        self.gate_proj = nn.Linear(self.embed_dim, self.ff_dim, bias=False)
        self.down_proj = nn.Linear(self.ff_dim, self.embed_dim, bias=False)

    def forward(self, x):
        up   = self.up_proj(x)
        gate = self.gate_proj(x)
        # GEGLU: GELU on the gate branch (tanh approximation used in Gemma3 impls)
        x = self.down_proj(F.gelu(gate, approximate="tanh") * up)
        return x

@dataclass
class KVCache:
    kx: torch.Tensor
    vx: torch.Tensor
    kv_cache_start_id: int
    kv_cache_end_id: int
    kv_cache_size: int

class Gemma3MHAttention(nn.Module):

    def __init__(self, max_length, embed_dim, num_heads, num_kv_heads, head_dim, is_sliding):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.max_length = max_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = (num_heads // num_kv_heads)
        self.is_sliding = is_sliding

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)

        self.q_norm = Gemma3RMSNorm(head_dim, eps=1e-6)
        self.k_norm = Gemma3RMSNorm(head_dim, eps=1e-6)

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)

        if self.is_sliding: self.window_size, self.rope_theta = 512, 10000.0
        else: self.rope_theta = 1_000_000.0

    @staticmethod
    def _rope_cos_sin(head_dim, rope_theta, position_ids, device, dtype):
        # θ_j = 1 / base^(2j/d) for j = 0, 1, ..., d/2 - 1
        theta_j = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float) / head_dim))

        # The is inverse to freq_j ∝ j (Logarithmially). So θ_j is proportional to wavelength.
        inv_freq_j = theta_j

        # Expand for batch/seq:
        inv_freq_j = inv_freq_j[None, :, None].expand(position_ids.shape[0], -1, 1).to(device)
        # [B, Hdim/2, 1]

        m_ids = position_ids[:, None, :].float()
        # [B, 1, T]

        # Compute cos/sin
        freqs = (inv_freq_j @ m_ids).transpose(1, 2)
        # [B, T, Hdim/2]

        emb = torch.cat((freqs, freqs), dim=-1)
        # [B, T, Hdim]

        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)

        return cos, sin
    
    @staticmethod
    def _rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        # This inserts the heads dimension so they can broadcast with q, k.
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        # Even tho q has H heads and k has Hkv heads. This works due to broadcast.
        # [B, 1, T, Hdim]
        q_embed = (q * cos) + (Gemma3MHAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (Gemma3MHAttention._rotate_half(k) * sin)
        return q_embed, k_embed
    
    def forward(self, x, past_kv=None, use_cache=False):
        H, Hkv = self.num_heads, self.num_kv_heads
        Demb, Dh = self.embed_dim, self.head_dim
        G = self.group_size
        B, T_q, _ = x.size()

        qx = self.q_proj(x)
        kx = self.k_proj(x)
        vx = self.v_proj(x)

        qx = qx.view(B, T_q, H,   Dh).permute(0, 2, 1, 3)      # [B, H,   T_q, Dh]
        kx = kx.view(B, T_q, Hkv, Dh).permute(0, 2, 1, 3)      # [B, Hkv, T_q, Dh]
        vx = vx.view(B, T_q, Hkv, Dh).permute(0, 2, 1, 3)      # [B, Hkv, T_q, Dh]

        kv_cache_start_id = 0 if past_kv is None else past_kv.kv_cache_start_id
        kv_cache_end_id = 0 if past_kv is None else past_kv.kv_cache_end_id
        kv_cache_size = 0 if past_kv is None else past_kv.kv_cache_size

        position_ids = torch.arange(kv_cache_end_id, kv_cache_end_id + T_q, device=qx.device)     # [T_q]
        position_ids = position_ids.unsqueeze(0).expand(B, -1)                                    # [B, T_q]

        qx = self.q_norm(qx)   # RMSNorm over Dh
        kx = self.k_norm(kx)

        cos, sin = Gemma3MHAttention._rope_cos_sin(Dh, self.rope_theta, position_ids=position_ids, device=qx.device, dtype=qx.dtype)
        qx, kx = Gemma3MHAttention.apply_rotary_pos_emb(qx, kx, cos, sin)
        # [B, H, T_q, Dh], [B, Hkv, T_q, Dh]

        if past_kv is not None:
            kp, vp = past_kv.kx, past_kv.vx                                 # [B, Hkv, kv_cache_size        Dh]
            Kx = torch.cat([kp, kx], dim=2)                                 # [B, Hkv, kv_cache_size + T_q, Dh]
            Vx = torch.cat([vp, vx], dim=2)                                 # [B, Hkv, kv_cache_size + T_q, Dh]
        else:
            Kx, Vx = kx, vx                                                 # [B, Hkv, T_q, Dh]

        qx = qx * (self.head_dim ** -0.5)
        # queries scaling just before attn

        qg = qx.reshape(B, Hkv, G, T_q, Dh)
        # [B, H, T_q, Dh] -> [B, Hkv, G, T_q, Dh]

        att_w = torch.einsum('bhgtd,bhkd->bhgtk', [qg.float(), Kx.float()])
        # [B, Hkv, G, T_q, Dh] @ [B, Hkv, kv_cache_size+T_q, Dh] = 
        # [B, Hkv, G, T_q, kv_cache_size+T_q]

        # Oh no we attended to every toks?
        # alternatively,
        query_positions = kv_cache_end_id + torch.arange(T_q, device=qx.device).view(T_q, 1)                                        # (T_q, 1)
        key_positions   = torch.arange(kv_cache_start_id, kv_cache_end_id+T_q, device=qx.device).view(1, kv_cache_size+T_q)         # (1, kv_cache_size+T_q)
        dist = query_positions - key_positions
        attn_mask = (dist >= 0)                                                                                                     # (T_q, kv_cache_size+T_q)
        if self.is_sliding:
            attn_mask = attn_mask & (dist <= self.window_size)
        attn_mask = attn_mask.view(1, 1, 1, T_q, kv_cache_size+T_q)
        # (1, 1, 1, T_q, kv_cache_size+T_q)

        att_w = att_w.masked_fill(~attn_mask, torch.finfo(att_w.dtype).min)
        att_w = F.softmax(att_w, dim=-1).to(qx.dtype)
        # [B, Hkv, G, T_q, kv_cache_size+T_q]

        ctx = torch.einsum('bhgtk,bhkd->bhgtd', [att_w, Vx])
        # [B, Hkv, G, T_q, kv_cache_size+T_q] @ [B, Hkv, kv_cache_size+T_q, Dh] = 
        # [B, Hkv, G, T_q, Dh]

        out = ctx.reshape(B, H, T_q, Dh)
        # [B, H, T_q, Dh]

        out = out.permute(0, 2, 1, 3).contiguous()
        # [B, T_q, H, Dh]

        out = self.o_proj(out.reshape(B, T_q, H * Dh))
        # [B, T_q, Demb]

        if use_cache:
            if self.is_sliding and Kx.size(2) > self.window_size:
                return out, KVCache(
                    kx=Kx[:, :, -self.window_size:, :],
                    vx=Vx[:, :, -self.window_size:, :],
                    kv_cache_start_id=kv_cache_start_id+T_q,
                    kv_cache_end_id=kv_cache_end_id+T_q,
                    kv_cache_size=self.window_size
                )
            
            else:
                return out, KVCache(
                    kx=Kx,
                    vx=Vx,
                    kv_cache_start_id=kv_cache_start_id,
                    kv_cache_end_id=kv_cache_end_id+T_q,
                    kv_cache_size=kv_cache_size+T_q
                )
               
        return out
        
class Gemma3Block(nn.Module):
    def __init__(self, max_length, embed_dim, ff_dim, num_heads, num_kv_heads, head_dim, is_sliding):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_sliding = is_sliding

        self.mhattn = Gemma3MHAttention(self.max_length, self.embed_dim, self.num_heads, self.num_kv_heads, self.head_dim, is_sliding)
        self.mlpf = Gemma3MLPF(self.embed_dim, self.ff_dim)

        self.pre_attn_norm = Gemma3RMSNorm(embed_dim, eps=1e-6)
        self.pos_attn_norm = Gemma3RMSNorm(embed_dim, eps=1e-6)
        self.pre_mlpf_norm = Gemma3RMSNorm(embed_dim, eps=1e-6)
        self.pos_mlpf_norm = Gemma3RMSNorm(embed_dim, eps=1e-6)

    def forward(self, x, past_kv=None, use_cache=False):
        if use_cache:
            attd_x, present_kv = self.mhattn(self.pre_attn_norm(x), past_kv=past_kv, use_cache=True)
            x = x + self.pos_attn_norm(attd_x)
            x = x + self.pos_mlpf_norm(self.mlpf(self.pre_mlpf_norm(x)))
            return x, present_kv
        else:
            x = x + self.pos_attn_norm(self.mhattn(self.pre_attn_norm(x)))
            x = x + self.pos_mlpf_norm(self.mlpf(self.pre_mlpf_norm(x)))
            return x
        
class Gemma3(nn.Module):

    def __init__(self, vocab_size, max_length, embed_dim, ff_dim, num_heads, num_kv_heads, head_dim, attentions):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length # this is not used anywhere tho due to rope scaling (gpt2 used to create fixed embeddings)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.attentions = attentions

        self.embedding = Gemma3Embedding(self.vocab_size, self.embed_dim)
        self.lm_head = Gemma3LMHead(self.embed_dim, self.embedding) # sending embedding to tie weights

        self.blocks = nn.ModuleList(
            Gemma3Block(
                max_length, embed_dim, ff_dim, num_heads, num_kv_heads, head_dim, is_sliding=True if t == 1 else False
            )
            for t in self.attentions
        )

    def forward(self, x, past_key_values=None, use_cache=False):
        # [B, SEQ_LENGTH]
        x = self.embedding(x)
        # [B, SEQ_LENGTH, EMBED_DIM]

        if use_cache:
            if past_key_values is None:
                past_key_values = [None] * len(self.attentions)
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
    def from_pretrained(cls, model_type='BASE', *, torch_dtype=torch.bfloat16, device='cuda'):

        assert model_type in ('BASE', 'INSTRUCT'), "only supports BASE, and INSTRUCT"

        assert device == 'cuda' and torch_dtype == torch.bfloat16, "not really tested otherwise due to gpu poor"

        attentions = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]
        config = dict(vocab_size=262144, max_length=32768, embed_dim=640, ff_dim=2048, num_heads=4, num_kv_heads=1, head_dim=256, attentions=attentions)
        
        MODEL_URL, DIR = {
            "BASE": ("https://huggingface.co/google/gemma-3-270m/resolve/main/model.safetensors", "gemma-3-270m"),
            "INSTRUCT": ("https://huggingface.co/google/gemma-3-270m-it/resolve/main/model.safetensors", "gemma-3-270m-instruct"),
        }[model_type]

        from utils import download_safetensors, stream_safetensors_to_meta_model

        # 1. safetensor version of the model
        model_file = download_safetensors(MODEL_URL, DIR)

        # 2. meta nn module
        with torch.device('meta'):
            model = cls(**config)

        # 3. mappings
        all_mappings = {
            'model.embed_tokens.weight': 'embedding.tok_embed.weight',
            'model.norm.weight':         'lm_head.norm.weight'
        }

        blocks_map = lambda i: {
            f'model.layers.{i}.input_layernorm.weight':             f'blocks.{i}.pre_attn_norm.weight',
            f'model.layers.{i}.post_attention_layernorm.weight':    f'blocks.{i}.pos_attn_norm.weight',
            f'model.layers.{i}.pre_feedforward_layernorm.weight':   f'blocks.{i}.pre_mlpf_norm.weight',
            f'model.layers.{i}.post_feedforward_layernorm.weight':  f'blocks.{i}.pos_mlpf_norm.weight',
            f'model.layers.{i}.mlp.up_proj.weight':                 f'blocks.{i}.mlpf.up_proj.weight',
            f'model.layers.{i}.mlp.gate_proj.weight':               f'blocks.{i}.mlpf.gate_proj.weight',
            f'model.layers.{i}.mlp.down_proj.weight':               f'blocks.{i}.mlpf.down_proj.weight',
            f'model.layers.{i}.self_attn.q_proj.weight':            f'blocks.{i}.mhattn.q_proj.weight',
            f'model.layers.{i}.self_attn.q_norm.weight':            f'blocks.{i}.mhattn.q_norm.weight',
            f'model.layers.{i}.self_attn.k_proj.weight':            f'blocks.{i}.mhattn.k_proj.weight',
            f'model.layers.{i}.self_attn.k_norm.weight':            f'blocks.{i}.mhattn.k_norm.weight',
            f'model.layers.{i}.self_attn.v_proj.weight':            f'blocks.{i}.mhattn.v_proj.weight',
            f'model.layers.{i}.self_attn.o_proj.weight':            f'blocks.{i}.mhattn.o_proj.weight',
        }
        for i in range(len(config['attentions'])):
            all_mappings.update(blocks_map(i))

        # 4. if any weights needs T
        needs_T = []

        model = stream_safetensors_to_meta_model(model, model_file, all_mappings, needs_T, torch_dtype, device)

        # Tokenizer init:
        tokenizer = Gemma3HFTokenizer.gemma3()

        if model_type == 'BASE':
            tokenizer.encode = tokenizer._encode
        elif model_type == 'INSTRUCT':
            tokenizer.encode = tokenizer._encode_instruct

        return tokenizer, model