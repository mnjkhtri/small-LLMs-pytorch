import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.infer import HookPoint, InferenceMixin
from models.pretrained import FromPretrainedMixin
from models.llama3.spec import LLAMA3_SPECS


class Llama3Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.W_E = nn.Embedding(self.vocab_size, self.d_model)
        self.hook_resid_emb = HookPoint()

    @property
    def W_E_weight(self):
        return self.W_E.weight

    def forward(self, toks):
        resid = self.W_E(toks)
        resid = self.hook_resid_emb(resid)
        return resid
    

class Llama3LMHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.ln_f = nn.RMSNorm(self.d_model, eps=1e-5)

    def forward(self, resid, W_U):
        resid = self.ln_f(resid)
        logits = F.linear(resid, W_U)
        return logits


class Llama3MLPF(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.up_proj = nn.Linear(self.d_model, self.d_mlp, bias=False)
        self.gate_proj = nn.Linear(self.d_model, self.d_mlp, bias=False)
        self.down_proj = nn.Linear(self.d_mlp, self.d_model, bias=False)

    def forward(self, resid):
        up = self.up_proj(resid)
        gate = self.gate_proj(resid)
        return self.down_proj(up * F.silu(gate))
    

class Llama3MHAttention(nn.Module):
    def __init__(self, max_length, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.max_length = max_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        self.group_size = n_heads // n_kv_heads

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.hook_attn_q = HookPoint()
        self.hook_attn_k = HookPoint()
        self.hook_attn_v = HookPoint()
        self.hook_attn_scores = HookPoint()
        self.hook_attn_probs = HookPoint()

        self.register_buffer("mask", torch.empty(0, dtype=torch.bool), persistent=False)

        self.rope = {
            "rope_theta": 500000.0,
            "factor": 32.0,
            "hi_freq_factor": 4.0,
            "lo_freq_factor": 1.0,
            "original_context_length": 8192,
        }

    def _get_mask(self, Tp, Tt, device):
        if Tt <= self.max_length:
            if self.mask.numel() == 0:
                m = torch.tril(torch.ones(self.max_length, self.max_length, device=device, dtype=torch.bool))
                self.mask = m.view(1, 1, self.max_length, self.max_length)
            return self.mask[:, :, Tp:Tt, :Tt]
        q = torch.arange(Tp, Tt, device=device)[:, None]
        k = torch.arange(0, Tt, device=device)[None, :]
        m = (k <= q)
        return m.view(1, 1, Tt - Tp, Tt)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _rope_cos_sin(head_dim, rope, x_pos, device, dtype):
        idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float) / head_dim
        freq = (1.0 / (2 * math.pi)) * (rope["rope_theta"] ** (-idx))
        # cycles/tok, in [1 / (TAU * rope_theta), 1 / TAU]
        # [TAU toks, TAU * rope_theta]
        factor = rope["factor"]
        lo = rope["lo_freq_factor"]
        hi = rope["hi_freq_factor"]
        L0 = rope["original_context_length"]
        freq_low  = lo / L0
        freq_high = hi / L0
        freq_scaled = torch.where(freq < freq_low, freq / factor, freq)
        smooth = (L0 * freq - lo) / (hi - lo)
        smooth = smooth.clamp(0.0, 1.0)
        freq_smooth = (1.0 - smooth) * (freq / factor) + smooth * freq
        is_mid = (freq >= freq_low) & (freq <= freq_high)
        freq = torch.where(is_mid, freq_smooth, freq_scaled)
        freq = freq[None, :, None].expand(x_pos.shape[0], -1, 1).to(device)
        phase = 2 * math.pi * (freq @ x_pos[:, None, :].float()).transpose(1, 2)  # cycles
        emb = torch.cat((phase, phase), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q = (q * cos) + (Llama3MHAttention._rotate_half(q) * sin)
        k = (k * cos) + (Llama3MHAttention._rotate_half(k) * sin)
        return q, k

    def _split_heads_q(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def _split_heads_kv(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x):
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(self, resid, past_kv=None, use_cache=False):
        B, T_new, _ = resid.size()                                    # resid: [B, T_new, d_model]

        q = self._split_heads_q(self.q_proj(resid))                   # q: [B, H,   T_new, d_head]
        k = self._split_heads_kv(self.k_proj(resid))                  # k: [B, Hkv, T_new, d_head]
        v = self._split_heads_kv(self.v_proj(resid))                  # v: [B, Hkv, T_new, d_head]

        q = self.hook_attn_q(q)
        k = self.hook_attn_k(k)
        v = self.hook_attn_v(v)

        K_past, V_past = past_kv if past_kv is not None else (None, None)
        T_past = 0 if K_past is None else K_past.size(2)

        x_pos = torch.arange(T_past, T_past + T_new, dtype=resid.dtype, device=resid.device).unsqueeze(0).expand(B, -1)
        cos, sin = self._rope_cos_sin(self.d_head, self.rope, x_pos=x_pos, device=resid.device, dtype=resid.dtype)
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        K = k if K_past is None else torch.cat([K_past, k], dim=2)    # K: [B, Hkv, T_total, d_head]
        V = v if V_past is None else torch.cat([V_past, v], dim=2)    # V: [B, Hkv, T_total, d_head]
        T_total = T_past + T_new

        # Grouped-query attention: split H query heads into Hkv groups of size G (H = Hkv * G)
        qg = q.reshape(B, self.n_kv_heads, self.group_size, T_new, self.d_head)
        # qg: [B, Hkv, G, T_new, d_head]

        att_g = torch.einsum("bhgqd,bhkd->bhgqk", qg.float(), K.float()) / (self.d_head ** 0.5)
        # att_g: [B, Hkv, G, T_new, T_total]

        attn_mask = self._get_mask(T_past, T_total, resid.device).unsqueeze(2)
        # att_g = att_g.masked_fill(~attn_mask, torch.finfo(att_g.dtype).min)
        att_g = att_g.masked_fill(~attn_mask, float("-inf"))

        att_scores = att_g.reshape(B, self.n_heads, T_new, T_total)   # [B, H, T_new, T_total]
        att_scores = self.hook_attn_scores(att_scores)
        att_g = att_scores.reshape(B, self.n_kv_heads, self.group_size, T_new, T_total)

        att_g = F.softmax(att_g, dim=-1).to(q.dtype)                  # [B, Hkv, G, T_new, T_total]

        att_probs = att_g.reshape(B, self.n_heads, T_new, T_total)    # [B, H, T_new, T_total]
        att_probs = self.hook_attn_probs(att_probs)
        att_g = att_probs.reshape(B, self.n_kv_heads, self.group_size, T_new, T_total)

        ctx = torch.einsum("bhgqk,bhkd->bhgqd", att_g, V)             # ctx: [B, Hkv, G, T_new, d_head]

        out = ctx.reshape(B, self.n_heads, T_new, self.d_head)        # out: [B, H, T_new, d_head]
        y = self.o_proj(self._merge_heads(out))                       # y: [B, T_new, d_model]

        return (y, (K, V)) if use_cache else y
    

class Llama3Block(nn.Module):
    def __init__(self, max_length, d_model, d_mlp, n_heads, n_kv_heads):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        self.mhattn = Llama3MHAttention(self.max_length, self.d_model, self.n_heads, self.n_kv_heads)
        self.mlpf = Llama3MLPF(self.d_model, self.d_mlp)

        self.ln1 = nn.RMSNorm(self.d_model, eps=1e-5)
        self.ln2 = nn.RMSNorm(self.d_model, eps=1e-5)

        self.hook_resid_pre = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, resid, past_kv=None, use_cache=False):
        resid = self.hook_resid_pre(resid)

        if use_cache:
            attn_out, present_kv = self.mhattn(self.ln1(resid), past_kv=past_kv, use_cache=True)
        else:
            attn_out = self.mhattn(self.ln1(resid))
            present_kv = None

        attn_out = self.hook_attn_out(attn_out)
        resid = resid + attn_out
        resid = self.hook_resid_mid(resid)

        mlp_out = self.mlpf(self.ln2(resid))
        mlp_out = self.hook_mlp_out(mlp_out)

        resid = resid + mlp_out
        resid = self.hook_resid_post(resid)

        return (resid, present_kv) if use_cache else resid


class Llama3(nn.Module, FromPretrainedMixin, InferenceMixin):
    PRETRAINED_SPECS = LLAMA3_SPECS

    def __init__(self, vocab_size, max_length, embed_dim, ff_dim, num_heads, num_kv_heads, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.d_model = embed_dim
        self.d_mlp = ff_dim
        self.n_heads = num_heads
        self.n_kv_heads = num_kv_heads
        self.n_layers = num_layers

        self.embed = Llama3Embedding(self.vocab_size, self.d_model)
        self.lm_head = Llama3LMHead(self.d_model)

        self.blocks = nn.ModuleList(
            [Llama3Block(self.max_length, self.d_model, self.d_mlp, self.n_heads, self.n_kv_heads) for _ in range(self.n_layers)]
        )

    def forward(self, toks, past_key_values=None, use_cache=False):
        resid = self.embed(toks)

        if use_cache and past_key_values is None:
            past_key_values = [None] * self.n_layers

        new_past_key_values = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            if use_cache:
                resid, present_kv = block(resid, past_kv=past_key_values[i], use_cache=True)
                new_past_key_values.append(present_kv)
            else:
                resid = block(resid)

        logits = self.lm_head(resid, self.embed.W_E_weight)
        out = {"logits": logits}
        if use_cache:
            out["past_key_values"] = new_past_key_values
        return out
