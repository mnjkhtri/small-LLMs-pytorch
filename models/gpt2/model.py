import torch
import torch.nn as nn
import torch.nn.functional as F

from models.infer import HookPoint, InferenceMixin
from models.pretrained import FromPretrainedMixin
from models.gpt2.spec import GPT2_SPECS


class GPT2Embedding(nn.Module):
    def __init__(self, vocab_size, max_length, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.d_model = d_model

        # W_E: [V, De]
        self.W_E = nn.Embedding(self.vocab_size, self.d_model)

        # W_pos: [L, De]
        self.W_pos = nn.Embedding(self.max_length, self.d_model)

        self.hook_resid_emb = HookPoint()

    @property
    def W_E_weight(self):
        return self.W_E.weight

    def forward(self, toks, pos):
        # toks : [B, T]
        # pos  : [1, T]
        tok = self.W_E(toks)      # [B, T, De]
        pos = self.W_pos(pos)     # [1, T, De]
        resid = tok + pos         # [B, T, De]
        resid = self.hook_resid_emb(resid)
        return resid


class GPT2LMHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.ln_f = nn.LayerNorm(self.d_model)

    def forward(self, resid, W_U):
        # resid : [B, T, De]
        # W_U   : [V, De]
        resid = self.ln_f(resid) # [B, T, De]
        logits = F.linear(resid, W_U) # [B, T, V]
        return logits


class GPT2(nn.Module, FromPretrainedMixin, InferenceMixin):
    PRETRAINED_SPECS = GPT2_SPECS

    def __init__(self, vocab_size, max_length, d_model, d_mlp, n_heads, n_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.embed = GPT2Embedding(self.vocab_size, self.max_length, self.d_model)
        self.lm_head = GPT2LMHead(self.d_model)

        self.blocks = nn.ModuleList(
            [GPT2Block(self.max_length, self.d_model, self.d_mlp, self.n_heads) for _ in range(self.n_layers)]
        )

    def forward(self, toks, past_key_values=None, use_cache=False):
        """
        Autoregressive rule:
            token t must be generated using information from tokens 0..t-1 (the prefix).

        When we generate step-by-step, the prefix does not change between steps; it only grows.
        So whatever the model already computed about the prefix can be stored and reused.

        past_key_values is that reusable "attention memory".

        For each layer:
            past_key_values[layer] = (K_past, V_past)

        Think:
            each past token contributes one key vector and one value vector (per head),
            and as we generate new tokens, this memory grows by appending new (K, V).
        """
        # toks: [B, T_new]
        B, T_new = toks.size()

        # prefix length already stored in memory
        if past_key_values is not None and past_key_values[0] is not None:
            T_past = past_key_values[0][0].size(2)  # K_past: [B, H, T_past, Dh]
        else:
            T_past = 0

        T_total = T_past + T_new
        assert T_total <= self.max_length, "sequence length cant exceed max length"

        # positions for the new tokens start after the prefix
        pos = torch.arange(T_past, T_total, device=toks.device).unsqueeze(0) # [1, T_new]
        resid = self.embed(toks, pos=pos) # [B, T_new, De]

        if use_cache and past_key_values is None:
            past_key_values = [None] * self.n_layers # first generation step

        new_past_key_values = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            if use_cache:
                resid, present_kv = block(resid, past_kv=past_key_values[i], use_cache=True) # resid: [B,T_new,De]
                new_past_key_values.append(present_kv)
            else:
                resid = block(resid) # [B, T_new, De]

        logits = self.lm_head(resid, self.embed.W_E_weight) # [B, T_new, V]

        out = {"logits": logits}
        if use_cache:
            out["past_key_values"] = new_past_key_values
        return out


class GPT2Block(nn.Module):
    def __init__(self, max_length, d_model, d_mlp, n_heads):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.n_heads = n_heads

        self.mhattn = GPT2MHAttention(self.max_length, self.d_model, self.n_heads)
        self.mlpf = GPT2MLPF(self.d_model, self.d_mlp)

        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

        self.hook_resid_pre = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, resid, past_kv=None, use_cache=False):
        # resid: [B, T, De]

        resid = self.hook_resid_pre(resid)

        if use_cache:
            attn_out, present_kv = self.mhattn(self.ln1(resid), past_kv=past_kv, use_cache=True) # [B,T,De]
        else:
            attn_out = self.mhattn(self.ln1(resid)) # [B,T,De]
            present_kv = None

        attn_out = self.hook_attn_out(attn_out)

        resid = resid + attn_out  # [B, T, De]
        resid = self.hook_resid_mid(resid)

        mlp_out = self.mlpf(self.ln2(resid)) # [B, T, De]
        mlp_out = self.hook_mlp_out(mlp_out)

        resid = resid + mlp_out # [B, T, De]
        resid = self.hook_resid_post(resid)

        return (resid, present_kv) if use_cache else resid


class GPT2MLPF(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp

        self.W_in = nn.Linear(self.d_model, self.d_mlp)   # [B,T,De] -> [B,T,Df]
        self.W_out = nn.Linear(self.d_mlp, self.d_model)  # [B,T,Df] -> [B,T,De]

    def forward(self, resid):
        # resid: [B, T, De]
        resid = self.W_in(resid)                       # [B, T, Df]
        resid = F.gelu(resid, approximate="tanh")      # [B, T, Df]
        resid = self.W_out(resid)                      # [B, T, De]
        return resid


class GPT2MHAttention(nn.Module):
    def __init__(self, max_length, d_model, n_heads):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0, "embed_dim must be divisible by num_heads"
        self.d_head = d_model // n_heads

        self.W_QKV = nn.Linear(self.d_model, 3 * self.d_model)  # [B,T,De] -> [B,T,3De]
        self.W_O = nn.Linear(self.d_model, self.d_model)        # [B,T,De] -> [B,T,De]

        self.hook_attn_q = HookPoint()
        self.hook_attn_k = HookPoint()
        self.hook_attn_v = HookPoint()
        self.hook_attn_scores = HookPoint()
        self.hook_attn_probs = HookPoint()

        self.register_buffer("mask", torch.empty(0, dtype=torch.bool), persistent=False)

    def _get_mask(self, Tp, Tt, device):
        if self.mask.numel() == 0:
            m = torch.tril(torch.ones(self.max_length, self.max_length, device=device, dtype=torch.bool))
            self.mask = m.view(1, 1, self.max_length, self.max_length)
        return self.mask[:, :, Tp:Tt, :Tt]

    def _split_heads(self, x):
        # x: [B, T, De] -> [B, H, T, Dh]
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        # x: [B, H, T, Dh] -> [B, T, De]
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(self, resid, past_kv=None, use_cache=False):
        """
        resid  : [B, T_new, De]
        past_kv: (K_past, V_past) where each is [B, H, T_past, Dh]
        return : y = [B, T_new, De]
                 if use_cache: also return (K_total, V_total) with T_total = T_past + T_new
        """
        B, T_new, _ = resid.size()

        qkv = self.W_QKV(resid)                           # [B, T_new, 3De]
        q, k, v = qkv.split(self.d_model, dim=2)          # each [B, T_new, De]
        q, k, v = map(self._split_heads, (q, k, v))       # each [B, H, T_new, Dh]

        q = self.hook_attn_q(q)
        k = self.hook_attn_k(k)
        v = self.hook_attn_v(v)

        K_past, V_past = past_kv if past_kv is not None else (None, None)
        T_past = 0 if K_past is None else K_past.size(2)

        K = k if K_past is None else torch.cat([K_past, k], dim=2)  # [B, H, T_total, Dh]
        V = v if V_past is None else torch.cat([V_past, v], dim=2)  # [B, H, T_total, Dh]
        T_total = T_past + T_new

        attn_mask = self._get_mask(T_past, T_total, resid.device) # [1, 1, T_new, T_total]

        att = torch.einsum("bhqd,bhkd->bhqk", q, K) / (self.d_head ** 0.5)  # [B, H, T_new, T_total]
        att = att.masked_fill(~attn_mask, torch.finfo(att.dtype).min)
        att = self.hook_attn_scores(att)
        att = F.softmax(att, dim=-1)
        att = self.hook_attn_probs(att)

        out = torch.einsum("bhqk,bhkd->bhqd", att, V)  # [B, H, T_new, Dh]
        y = self.W_O(self._merge_heads(out)) # [B, T_new, De]

        return (y, (K, V)) if use_cache else y
