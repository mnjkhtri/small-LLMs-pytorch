import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT2Embedding(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim

        self.tok_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_embed = nn.Embedding(self.max_length, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    @property
    def weight(self): # convenience alias
        return self.tok_embed.weight

    def forward(self, x, x_pos):
        # [B, SEQ_LENGTH]
        x = self.tok_embed(x) + self.pos_embed(x_pos) # (B, T, D) + (1, T, D): (B, T, D)
        # [B, SEQ_LENGTH, EMBED_DIM]
        return self.dropout(x)
        # Note: Essentially a table lookup of id to dense representation. Hence a huge matrix

class GPT2LMHead(nn.Module):
    def __init__(self, embed_dim, embedding):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.embedding = embedding

    def forward(self, x):
        # [B, SEQ_LENGTH, EMBED_DIM]
        w = self.embedding.weight
        x = self.ln(x)
        out = F.linear(x, w)
        return out
        # [B, SEQ_LENGTH, VOCAB_SIZE]
        
class MHAttention(nn.Module):
    def __init__(self, max_length, embed_dim, num_heads, dropout):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = (embed_dim // num_heads)

        # fused qkv projection:
        self.qkv_w = nn.Linear(self.embed_dim, 3*self.embed_dim)
        self.proj_w = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        """
        x: [B, T_q, C], T_q = 1 for inference (usually)
        past_kv: (K_cache, V_cache) where K_cache, V_cache: [B, H, T_past, Dh]
        use_cache: if True, return (y, (K_total, V_total)); else return y
        """

        B, T_q, _ = x.size()
        x = self.qkv_w(x)
        qx, kx, vx = x.split(self.embed_dim, dim=2)

        # each [B, H, T_q, Hd]:
        qx = qx.view(B, T_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kx = kx.view(B, T_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        vx = vx.view(B, T_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        mask = torch.tril(torch.ones(self.max_length, self.max_length, dtype=torch.bool)).view(1, 1, self.max_length, self.max_length).to(x.device)

        if past_kv is not None:
            kp, vp = past_kv                                                # [B, H, T_past, Dh]
            T_past = kp.size(2)
            Kx = torch.cat([kp, kx], dim=2)                                 # [B, H, T_past + T_q, Dh]
            Vx = torch.cat([vp, vx], dim=2)                                 # [B, H, T_past + T_q, Dh]
            attn_mask = mask[:, :, T_past:T_past+T_q, :T_past+T_q]     # [1, 1, T_q, T_past + T_q]
        else:
            Kx, Vx = kx, vx
            T_past = 0
            attn_mask = mask[:, :, :T_q, :T_q]                         # [1, 1, T_q, T_q]

        att_w = torch.einsum('bhqd,bhkd->bhqk', [qx, Kx]) / (self.head_dim ** 0.5)
        # [B, H, T_q, T_past + T_q]
        att_w = att_w.masked_fill(~attn_mask, torch.finfo(att_w.dtype).min)
        # [B, H, T_q, T_past + T_q]

        att_w = F.softmax(att_w, dim=-1)
        att_w = self.dropout(att_w)
        # [B, H, T_q, T_past + T_q]
        # droppin out probs why? (attn dropout)

        out = torch.einsum('bhal,bhlv->bhav', [att_w, Vx])
        # [B, H, T_q, Dh]
        out = out.permute(0,2,1,3).contiguous()
        # [B, T_q, H, Dh]
        out = out.view(B, -1, self.num_heads * self.head_dim)
        # [B, T_q, Edim]
        out = self.dropout(self.proj_w(out)) 
        # residue dropout

        if use_cache:
            return out, (Kx, Vx)
        
        return out
    
        # Note: Attention weights [T,T] is probability distribution over all tokens (summing to 1), telling how much to borrow from each.
        # Each output feature is a weighted average of the same feature across all tokens’ value vectors.
        # Head divides the feature space vertically.
        # Each head (with its own probability distribution) produces a weighted average of value vectors (features) within its subspace.
        # Here features within single head share a distribution.
        # Intuitively, token vector is updated as if it had a dynamically generated transformation matrix [E,E] applied to it.
        # As GNN: essentially can think of multiple parallel graphs, each head defining its own edge weights and message passing.
        # MHA = each node vertically split, and each slice attends only to corresponding slices in other tokens.

        # KV cache:
        # for each inference the KV cache grow as O(num of layers x seq length x num heads x head dim)

        # To empower kv cache inference, can does it make sense to alter the inductive bias of the model ie maybe multiple q heads share common kv head??

class MLPF(nn.Module):

    def __init__(self, embed_dim, ff_dim, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim

        self.emb_ff = nn.Linear(self.embed_dim, self.ff_dim)
        self.ff_emb = nn.Linear(self.ff_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.emb_ff(x)
        x = F.gelu(x, approximate="tanh")
        x = self.ff_emb(x)
        x = self.dropout(x)
        return x
        # Note: The affine transformaiton E -> E is not very expessive. By expanding four times and compressive back learns rich features.
        # This acts like a bottleneck autoencoder
        # In a stack 2/3rd of parameters live in MLPF while rest 1/3rd is in MHA

class GPT2Block(nn.Module):
    def __init__(self, max_length, embed_dim, ff_dim, num_heads, dropout):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads

        self.mhattn = MHAttention(self.max_length, self.embed_dim, self.num_heads, dropout)
        self.mlpf = MLPF(self.embed_dim, self.ff_dim, dropout)

        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x, past_kv=None, use_cache=False):
        if use_cache:
            attd_x, present_kv = self.mhattn(self.ln1(x), past_kv=past_kv, use_cache=True)
            x = x + attd_x
            x = x + self.mlpf(self.ln2(x))
            return x, present_kv
        else:
            x = x + self.mhattn(self.ln1(x))
            x = x + self.mlpf(self.ln2(x))
            return x

class GPT2(nn.Module):

    def __init__(self, vocab_size, max_length, embed_dim, ff_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = GPT2Embedding(self.vocab_size, self.max_length, self.embed_dim, dropout)
        self.lm_head = GPT2LMHead(self.embed_dim, self.embedding) # sending embedding to tie weights

        self.blocks = nn.ModuleList(
            [GPT2Block(max_length, embed_dim, ff_dim, num_heads, dropout) for _ in range(self.num_layers)]
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
        x_pos = torch.arange(T_past, T_past + T, device=x.device).unsqueeze(0)  # [1, T]
        x = self.embedding(x, x_pos=x_pos)
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
    def from_pretrained(cls, *, torch_dtype=torch.float32, device="cuda"):
        
        config = dict(vocab_size=50257, max_length=1024, embed_dim=768,  ff_dim=768*4,  num_heads=12, num_layers=12, dropout=0.1)
        model = cls(**config).to(dtype=torch_dtype, device=device)

        from .utils import download_safetensors, stream_safetensors_to_meta_model

        MODEL_URL = 'https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors'
        DIR = 'gpt2'

        # 1. model file
        model_file = download_safetensors(MODEL_URL, DIR)

        # 2. meta nn module
        with torch.device('meta'):
            model = cls(**config)

        # 3. embeddings
        embedding_map = {
            'wte.weight':      'embedding.tok_embed.weight',
            'wpe.weight':      'embedding.pos_embed.weight'
        }
        lm_head_map = {
            'ln_f.weight':     'lm_head.ln.weight',
            'ln_f.bias':       'lm_head.ln.bias'
        }
        blocks_map = lambda i: {
            f'h.{i}.ln_1.weight':           f'blocks.{i}.ln1.weight',
            f'h.{i}.ln_1.bias':             f'blocks.{i}.ln1.bias',
            f'h.{i}.attn.c_attn.weight':    f'blocks.{i}.mhattn.qkv_w.weight',
            f'h.{i}.attn.c_attn.bias':      f'blocks.{i}.mhattn.qkv_w.bias',
            f'h.{i}.attn.c_proj.weight':    f'blocks.{i}.mhattn.proj_w.weight',
            f'h.{i}.attn.c_proj.bias':      f'blocks.{i}.mhattn.proj_w.bias',
            f'h.{i}.ln_2.weight':           f'blocks.{i}.ln2.weight',
            f'h.{i}.ln_2.bias':             f'blocks.{i}.ln2.bias',
            f'h.{i}.mlp.c_fc.weight':       f'blocks.{i}.mlpf.emb_ff.weight',
            f'h.{i}.mlp.c_fc.bias':         f'blocks.{i}.mlpf.emb_ff.bias',
            f'h.{i}.mlp.c_proj.weight':     f'blocks.{i}.mlpf.ff_emb.weight',
            f'h.{i}.mlp.c_proj.bias':       f'blocks.{i}.mlpf.ff_emb.bias'
        }
        all_mappings = {**embedding_map, **lm_head_map}
        for i in range(config['num_layers']):
            all_mappings.update(blocks_map(i))

        # 4. needs T if any 
        needs_T = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        model = stream_safetensors_to_meta_model(model, model_file, all_mappings, needs_T, torch_dtype, device)
        return model
    