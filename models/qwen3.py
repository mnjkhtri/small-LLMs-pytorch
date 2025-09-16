from tokenizers import Tokenizer
from pathlib import Path
import requests
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3Tokenizers:
    def __init__(self, model_path):

        self.tokenizer = Tokenizer.from_file(model_path)
    
    def _encode(self, text):
        return self.tokenizer.encode(text).ids

    def _encode_instruct(self, user: str, system: str = "You are a helpful assistant."):
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        text = (
            f"{im_start}system\n{system}{im_end}\n"
            f"{im_start}user\n{user}{im_end}\n"
            f"{im_start}assistant\n"
        )
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    @classmethod
    def qwen3(cls):
        TOKENIZER_URL = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json"
        save_path = Path(".cache") / "qwen3-0.6b-instruct"
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
    
class Qwen3Embedding(nn.Module):
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

class Qwen3LMHead(nn.Module):
    def __init__(self, embed_dim, embedding):
        super().__init__()
        self.norm = nn.RMSNorm(embed_dim, eps=1e-6)
        self.embedding = embedding

    def forward(self, x):
        # [B, SEQ_LENGTH, EMBED_DIM]
        w = self.embedding.weight
        x = self.norm(x)
        out = F.linear(x, w)
        return out
        # [B, SEQ_LENGTH, VOCAB_SIZE]

class Qwen3MLPF(nn.Module):

    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim

        self.up_proj = nn.Linear(self.embed_dim, self.ff_dim, bias=False)
        self.gate_proj = nn.Linear(self.embed_dim, self.ff_dim, bias=False)
        self.down_proj = nn.Linear(self.ff_dim, self.embed_dim, bias=False)

    def forward(self, x):
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        x = self.down_proj(up * F.silu(gate))
        return x
        #  SwiGLU activation

class Qwen3MHAttention(nn.Module):

    def __init__(self, max_length, embed_dim, num_heads, num_kv_heads, head_dim):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.max_length = max_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = (num_heads // num_kv_heads)

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        # num_heads q share num_kv_heads kv;

        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)

        self.rope = {
            'rope_theta': 1_000_000.0
        }

    @staticmethod
    def _rope_cos_sin(head_dim, rope, position_ids, device, dtype):
        # θ_j = 1 / base^(2j/d) for j = 0, 1, ..., d/2 - 1
        theta_j = 1.0 / (rope['rope_theta'] ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float) / head_dim))

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
        q_embed = (q * cos) + (Qwen3MHAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (Qwen3MHAttention._rotate_half(k) * sin)
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

        T_past = 0 if past_kv is None else past_kv[0].size(2)

        position_ids = torch.arange(T_past, T_past + T_q, device=qx.device)   # [T_q]
        position_ids = position_ids.unsqueeze(0).expand(B, -1)                # [B, T_q]

        qx = self.q_norm(qx)   # RMSNorm over Dh
        kx = self.k_norm(kx)

        cos, sin = Qwen3MHAttention._rope_cos_sin(Dh, self.rope, position_ids=position_ids, device=qx.device, dtype=qx.dtype)
        qx, kx = Qwen3MHAttention.apply_rotary_pos_emb(qx, kx, cos, sin)
        # [B, H, T_q, Dh], [B, Hkv, T_q, Dh]

        if past_kv is not None:
            kp, vp = past_kv                                                # [B, Hkv, T_past,       Dh]
            Kx = torch.cat([kp, kx], dim=2)                                 # [B, Hkv, T_past + T_q, Dh]
            Vx = torch.cat([vp, vx], dim=2)                                 # [B, Hkv, T_past + T_q, Dh]
        else:
            Kx, Vx = kx, vx                                                 # [B, Hkv, T_q, Dh]

        attn_mask = torch.ones(T_q, T_past + T_q, dtype=torch.bool, device=qx.device).tril(T_past)
        attn_mask = attn_mask.view(1, 1, T_q, T_past + T_q)                 # [1, 1, T_q, T_q + T_past]
        # [1, 1, T_q, T_past + T_q]

        qg = qx.reshape(B, Hkv, G, T_q, Dh)
        # [B, H, T_q, Dh] -> [B, Hkv, G, T_q, Dh]

        att_w = torch.einsum('bhgtd,bhkd->bhgtk', [qg.float(), Kx.float()]) / (Dh ** 0.5)
        # [B, Hkv, G, T_q, Dh] @ [B, Hkv, T_past+T_q, Dh] = 
        # [B, Hkv, G, T_q, T_past+T_q]

        att_w = att_w.masked_fill(~attn_mask.unsqueeze(2), torch.finfo(att_w.dtype).min)
        att_w = F.softmax(att_w, dim=-1).to(qx.dtype)
        # [B, Hkv, G, T_q, T_past+T_q]

        ctx = torch.einsum('bhgtk,bhkd->bhgtd', [att_w, Vx])
        # [B, Hkv, G, T_q, T_past+T_q] @ [B, Hkv, T_past+T_q, Dh] = 
        # [B, Hkv, G, T_q, Dh]

        out = ctx.reshape(B, H, T_q, Dh)
        # [B, H, T_q, Dh]

        out = out.permute(0, 2, 1, 3).contiguous()
        # [B, T_q, H, Dh]

        out = self.o_proj(out.reshape(B, T_q, H * Dh))
        # [B, T_q, Demb]

        if use_cache:
            return out, (Kx, Vx)
        
        return out
        
class Qwen3Block(nn.Module):
    def __init__(self, max_length, embed_dim, ff_dim, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # wait wont embed_dim and num_heads fix the head_dim? no it doesnt in Qwen3

        self.mhattn = Qwen3MHAttention(self.max_length, self.embed_dim, self.num_heads, self.num_kv_heads, self.head_dim)
        self.mlpf = Qwen3MLPF(self.embed_dim, self.ff_dim)

        self.norm1 = nn.RMSNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(embed_dim, eps=1e-6)

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
        
class Qwen3(nn.Module):

    def __init__(self, vocab_size, max_length, embed_dim, ff_dim, num_heads, num_kv_heads, head_dim, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length # this is not used anywhere tho due to rope scaling (gpt2 used to create fixed embeddings)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers

        self.embedding = Qwen3Embedding(self.vocab_size, self.embed_dim)
        self.lm_head = Qwen3LMHead(self.embed_dim, self.embedding) # sending embedding to tie weights

        self.blocks = nn.ModuleList(
            [Qwen3Block(max_length, embed_dim, ff_dim, num_heads, num_kv_heads, head_dim) for _ in range(self.num_layers)]
        )

    def forward(self, x, past_key_values=None, use_cache=False):
        B, T = x.size()
        if past_key_values is not None and past_key_values[0] is not None:
            # K: [B, H, T_past, Dh]
            T_past = past_key_values[0][0].size(2)
        else:
            T_past = 0

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
    def from_pretrained(cls, model_type='BASE', *, torch_dtype=torch.bfloat16, device='cuda'):

        assert model_type in ('BASE', 'INSTRUCT'), "only supports BASE, and INSTRUCT"

        assert device == 'cuda' and torch_dtype == torch.bfloat16, "not really tested otherwise due to gpu poor"

        config = dict(vocab_size=151936, max_length=32768, embed_dim=1024, ff_dim=2736, num_heads=16, num_kv_heads=8, head_dim=128, num_layers=28)
        # head dim is not fixed by embed_dim and num_heads now. it is different, do divide and see

        MODEL_URL, DIR = {
            "BASE": ("https://huggingface.co/qwen/qwen3-0.6b-base/resolve/main/model.safetensors", "qwen3-0.6b"),
            "INSTRUCT": ("https://huggingface.co/qwen/qwen3-0.6b/resolve/main/model.safetensors", "qwen3-0.6b-instruct"),
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
            f'model.layers.{i}.input_layernorm.weight':           f'blocks.{i}.norm1.weight',
            f'model.layers.{i}.mlp.up_proj.weight':               f'blocks.{i}.mlpf.up_proj.weight',
            f'model.layers.{i}.mlp.gate_proj.weight':             f'blocks.{i}.mlpf.gate_proj.weight',
            f'model.layers.{i}.mlp.down_proj.weight':             f'blocks.{i}.mlpf.down_proj.weight',
            f'model.layers.{i}.self_attn.q_proj.weight':          f'blocks.{i}.mhattn.q_proj.weight',
            f'model.layers.{i}.self_attn.q_norm.weight':          f'blocks.{i}.mhattn.q_norm.weight',
            f'model.layers.{i}.self_attn.k_proj.weight':          f'blocks.{i}.mhattn.k_proj.weight',
            f'model.layers.{i}.self_attn.k_norm.weight':          f'blocks.{i}.mhattn.k_norm.weight',
            f'model.layers.{i}.self_attn.v_proj.weight':          f'blocks.{i}.mhattn.v_proj.weight',
            f'model.layers.{i}.self_attn.o_proj.weight':          f'blocks.{i}.mhattn.o_proj.weight',
            f'model.layers.{i}.post_attention_layernorm.weight':  f'blocks.{i}.norm2.weight',
        }
        for i in range(config['num_layers']):
            all_mappings.update(blocks_map(i))

        # 4. if any weights needs T
        needs_T = []

        model = stream_safetensors_to_meta_model(model, model_file, all_mappings, needs_T, torch_dtype, device)

        # Tokenizer init:
        tokenizer = Qwen3Tokenizers.qwen3()

        if model_type == 'BASE':
            tokenizer.encode = tokenizer._encode
        elif model_type == 'INSTRUCT':
            tokenizer.encode = tokenizer._encode_instruct

        return tokenizer, model