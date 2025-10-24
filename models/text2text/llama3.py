import tiktoken
from tiktoken.load import load_tiktoken_bpe
from pathlib import Path
import requests
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Llama3Tiktoken:
    def __init__(self, model_path, pat_str):

        mergeable_ranks = load_tiktoken_bpe(model_path)
        special_tokens = [
            "<|begin_of_text|>",                # BOS (beginning of sequence) – always added at start
            "<|end_of_text|>",                  # EOS (end of sequence) – marks end of document/output
            "<|reserved_special_token_0|>",     # placeholder, never trained (future expansion)
            "<|reserved_special_token_1|>",     # placeholder
            "<|reserved_special_token_2|>",     # placeholder
            "<|reserved_special_token_3|>",     # placeholder
            "<|start_header_id|>",              # used in instruct/chat models: marks start of role header (e.g., user/assistant/system)
            "<|end_header_id|>",                # used in instruct/chat models: marks end of role header
            "<|reserved_special_token_4|>",     # another placeholder, unused
            "<|eot_id|>",                       # end-of-turn marker (chat fine-tuning); every message ends with this
        ] + [
            # Remaining reserved slots: unused placeholders filling up the reserved 256-token block.
            f"<|reserved_special_token_{i}|>"
            for i in range(5, 256 - 5)
        ]
        self.special_tokens = {token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)}

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
    
    def _encode(self, text):
        return [self.special_tokens['<|begin_of_text|>']] + self.model.encode(text, allowed_special=set(), disallowed_special=set())

    def _encode_instruct(self, user: str, system: str = "You are a helpful assistant."):
        st = self.special_tokens
        enc = lambda s: self.model.encode(s, allowed_special=set(), disallowed_special=set())
        toks = [st['<|begin_of_text|>']]
        # System
        toks += [st['<|start_header_id|>']] + enc("system") + [st['<|end_header_id|>']]
        toks += enc("\n\n" + system) + [st['<|eot_id|>']]
        # User
        toks += [st['<|start_header_id|>']] + enc("user") + [st['<|end_header_id|>']]
        toks += enc("\n\n" + user) + [st['<|eot_id|>']]
        # Assistant (generation will continue from here)
        toks += [st['<|start_header_id|>']] + enc("assistant") + [st['<|end_header_id|>']]
        toks += enc("\n\n")
        return toks

    def decode(self, ids):
        return self.model.decode(ids)

    @classmethod
    def llama3(cls):
        TOKENIZER_URL = "https://huggingface.co/meta-llama/llama-3.2-1b-instruct/resolve/main/original/tokenizer.model"
        save_path = Path(".cache") / "llama-3.2-1b-instruct"
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

        pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        return cls(str(model_file), pat_str)


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

class Llama3MHAttention(nn.Module):

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

        self.rope = {
            "rope_theta": 500000.0,
            "factor": 32.0,
            "hi_freq_factor": 4.0,
            "lo_freq_factor": 1.0,
            "original_context_length": 8192
        }

    @staticmethod
    def _rope_cos_sin(
        head_dim,
        rope,
        position_ids,
        device,
        dtype,
    ):
        # θ_j = 1 / base^(2j/d) for j = 0, 1, ..., d/2 - 1
        theta_j = 1.0 / (rope['rope_theta'] ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float) / head_dim))

        # The is inverse to freq_j ∝ j (Logarithmially). So θ_j is proportional to wavelength.
        inv_freq_j = theta_j

        # Their values range geometrically from:
        #   θ_j=0       = 1                             (highest frequency, fastest subspace)
        #   θ_j=d/2     = 1 / rope_theta                (lowest frequency,  slowest subspace)
        # So: θ_j ∈ [1 / rope_theta, 1]

        # The actual rotation angle for position m (equivalent to time) is (will be):
        #   φ_j(m) = m * θ_j

        # Effect of increasing m (example with context_len = 8192):
        #
        # High-frequency subspaces (θ ≈ 1):
        # Each step in m increases φ by ~1 radian (cycle completes after 6.28 tokens, wavelen)
        # At m = 8192, φ ≈ 8192 radians
        # Since exp/cos/sin repeat every 2π ≈ 6.28 radians, that’s about 1304 full cycles
        # These subspaces oscillate rapidly, capturing fine local position differences
        #
        # Low-frequency subspaces (θ ≈ 1 / 500_000 = 0.000002):
        # Each step in m increases φ by only 0.000002 radians
        # At m = 8192, φ ≈ 0.016 radians (tiny angle, ~1/400 of a full turn) (cycle completes after 3,141,593 tokens)
        # Exp/cos/sin barely change at all across the context
        # These subspaces evolve extremely slowly, encoding broad, long-range position

        # what if context_len = 32k?
        # HF subspace: 5100 cycles vs LF subspace: (still tiny, << 1 cycle)
        # Positions beyond training length (e.g. 32k vs 8k) cause new aliasing patterns the model was never trained on
        # With scaling, the cos/sin signals look statistically similar to what the model saw in training, just stretched to cover the longer sequence

        # The solution: rescale θ_j by frequency band:
        # High-frequency bands (short λ_j): Left unchanged to preserve local detail
        # Low-frequency bands (long λ_j): Scaled down (slowed further) so they stretch smoothly over new context size
        # Medium-frequency bands: Smoothly interpolated between the two, avoiding sharp cutoffs

        factor = rope["factor"]
        lo_freq_factor = rope["lo_freq_factor"]
        hi_freq_factor = rope["hi_freq_factor"]
        original_context_length = rope["original_context_length"]

        wavelen = 2 * math.pi / inv_freq_j
        # the number of tokens it takes for cos/sin in that subspace to complete one full cycle (2π radians)

        wavelen_cutoff = (original_context_length / hi_freq_factor, original_context_length / lo_freq_factor)

        # wavelen < 2048 means freq is TOO HIGH: as it is.
        pass

        # wavelen > 8192 means freq is TOO LOW: need to scale it up
        inv_freq_j = torch.where(wavelen > wavelen_cutoff[1], inv_freq_j / factor, inv_freq_j)

        # wavelen in [2028, 8192] is interpolation:
        smooth_factor = (original_context_length / wavelen - lo_freq_factor) / (hi_freq_factor - lo_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_j / factor + smooth_factor * inv_freq_j
        is_medium_freq = ~(wavelen < wavelen_cutoff[1]) * ~(wavelen > wavelen_cutoff[0])
        inv_freq_j = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_j)

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
        q_embed = (q * cos) + (Llama3MHAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (Llama3MHAttention._rotate_half(k) * sin)
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

        cos, sin = Llama3MHAttention._rope_cos_sin(Dh, self.rope, position_ids=position_ids, device=qx.device, dtype=qx.dtype)
        qx, kx = Llama3MHAttention.apply_rotary_pos_emb(qx, kx, cos, sin)
        # [B, H, T_q, Dh], [B, Hkv, T_q, Dh]

        if past_kv is not None:
            kp, vp = past_kv                                                # [B, Hkv, T_past,       Dh]
            Kx = torch.cat([kp, kx], dim=2)                                 # [B, Hkv, T_past + T_q, Dh]
            Vx = torch.cat([vp, vx], dim=2)                                 # [B, Hkv, T_past + T_q, Dh]
        else:
            Kx, Vx = kx, vx                                                 # [B, Hkv, T_q, Dh]

        qg = qx.reshape(B, Hkv, G, T_q, Dh)
        # [B, H, T_q, Dh] -> [B, Hkv, G, T_q, Dh]

        att_w = torch.einsum('bhgtd,bhkd->bhgtk', [qg.float(), Kx.float()]) / (Dh ** 0.5)
        # [B, Hkv, G, T_q, Dh] @ [B, Hkv, T_past+T_q, Dh] = 
        # [B, Hkv, G, T_q, T_past+T_q]

        # Oh no we shouldnt have attended all the toks only casual ones?
        attn_mask = torch.ones(T_q, T_past + T_q, dtype=torch.bool, device=qx.device).tril(T_past).view(1, 1, 1, T_q, T_past+T_q)  # [1, 1, 1, T_q, T_past+T_q]
        att_w = att_w.masked_fill(~attn_mask, torch.finfo(att_w.dtype).min)
        att_w = F.softmax(att_w, dim=-1).to(qx.dtype)
        # [B, Hkv, G, T_q, T_past+T_q]

        ctx = torch.einsum('bhgtk,bhkd->bhgtd', [att_w, Vx])
        # [B, Hkv, G, T_q, T_past+T_q] @ [B, Hkv, T_past+T_q, Dh] = 
        # [B, Hkv, G, T_q, Dh]

        out = ctx.reshape(B, H, T_q, Dh)
        # [B, H, T_q, Dh]

        out = out.permute(0, 2, 1, 3).contiguous()
        # [B, T_q, H, Dh]
        out = out.reshape(B, T_q, Demb)
        # [B, T_q, Demb]

        out = self.o_proj(out)
        # [B, T_q, Demb]

        if use_cache:
            return out, (Kx, Vx)
        
        return out
        
class Llama3MLPF(nn.Module):

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

        self.mhattn = Llama3MHAttention(self.max_length, self.embed_dim, self.num_heads, self.num_kv_heads)
        self.mlpf = Llama3MLPF(self.embed_dim, self.ff_dim)

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
        self.max_length = max_length # this is not used anywhere tho due to rope scaling (gpt2 used to create fixed embeddings)
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

        config = dict(vocab_size=128256, max_length=8192, embed_dim=2048, ff_dim=4*2048, num_heads=32, num_kv_heads=8, num_layers=16)

        MODEL_URL, DIR = {
            "BASE": ("https://huggingface.co/meta-llama/llama-3.2-1b/resolve/main/model.safetensors", "llama-3.2-1b"),
            "INSTRUCT": ("https://huggingface.co/meta-llama/llama-3.2-1b-instruct/resolve/main/model.safetensors", "llama-3.2-1b-instruct")
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
            f'model.layers.{i}.self_attn.k_proj.weight':          f'blocks.{i}.mhattn.k_proj.weight',
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

        tokenizer = Llama3Tiktoken.llama3()

        if model_type == 'BASE':
            tokenizer.encode = tokenizer._encode
        elif model_type == 'INSTRUCT':
            tokenizer.encode = tokenizer._encode_instruct

        return tokenizer, model