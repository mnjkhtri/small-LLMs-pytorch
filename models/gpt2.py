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

    def forward(self, x):
        _, seq = x.size()
        # [B, SEQ_LENGTH]
        x_pos = torch.arange(seq, device=x.device).unsqueeze(0)
        x = self.tok_embed(x) + self.pos_embed(x_pos) # (B, T, D) + (1, T, D): (B, T, D)
        # [B, SEQ_LENGTH, EMBED_DIM]
        return self.dropout(x)

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

        # precreate a max mask and use necessary bits during forward
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.max_length, self.max_length, dtype=torch.bool))
            .view(1, 1, self.max_length, self.max_length),
            persistent=False
        )

    def forward(self, x):
        B, T, _ = x.size()
        x = self.qkv_w(x)
        qx, kx, vx = x.split(self.embed_dim, dim=2)
        qx = qx.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kx = kx.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        vx = vx.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        att_w = torch.einsum('bhqd,bhkd->bhqk', [qx, kx])/(self.head_dim ** 0.5)
        att_w = att_w.masked_fill(~self.mask[:, :, :T, :T], torch.finfo(att_w.dtype).min)
        att_w = self.dropout(F.softmax(att_w, dim=-1)) 
        # droppin out probs why? (attn dropout)
        out = torch.einsum('bhal,bhlv->bhav', [att_w, vx]).permute(0,2,1,3).contiguous()
        out = out.view(x.shape[0], -1, self.num_heads * self.head_dim)
        out = self.dropout(self.proj_w(out)) 
        # residue dropout
        return out
    
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

    def forward(self, x):
        # [B, SEQ_LENGTH, EMBED_DIM]
        x = x + self.mhattn(self.ln1(x))
        # [B, SEQ_LENGTH, EMBED_DIM]
        x = x + self.mlpf(self.ln2(x))
        # [B, SEQ_LENGTH, EMBED_DIM]
        return x

class GPT2LMHead(nn.Module):
    def __init__(self, embed_dim, vocab_size, embedding):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie weights: both modules now share the same nn.Parameter:
        self.lm_head.weight = embedding.tok_embed.weight

    def forward(self, x):
        # [B, SEQ_LENGTH, EMBED_DIM]
        x = self.ln(x)
        x = self.lm_head(x)
        return x
        # [B, SEQ_LENGTH, VOCAB_SIZE]

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
        self.lm_head = GPT2LMHead(self.embed_dim, self.vocab_size, self.embedding) # sending embedding to tie weights

        self.stack = nn.Sequential(*[GPT2Block(max_length, embed_dim, ff_dim, num_heads, dropout) for _ in range(self.num_layers)])


    def forward(self, x):
        B, T = x.size()
        assert T <= self.max_length, "sequence length cant exceed max length"

        # [B, SEQ_LENGTH]
        x = self.embedding(x)
        # [B, SEQ_LENGTH, EMBED_DIM]
        x = self.stack(x)
        # [B, SEQ_LENGTH, EMBED_DIM]
        x = self.lm_head(x)

        return x

    @classmethod
    def from_pretrained(cls, model_type, *, torch_dtype=torch.float32, device="cuda"):
        """
        1. materialize weights from HF on CPU
        2. allocate target parameters on GPU
        3. parameter mapping and per tensor transfer
        """
        
        assert model_type in ('gpt2', 'gpt2-medium'), "please use only gpt2 or gpt2-medium, we poor"
        config = {
            'gpt2':         dict(vocab_size=50257, max_length=1024, embed_dim=768,  ff_dim=768*4,  num_heads=12, num_layers=12, dropout=0.1),
            'gpt2-medium':  dict(vocab_size=50257, max_length=1024, embed_dim=1024, ff_dim=1024*4, num_heads=16, num_layers=24, dropout=0.1)
        }[model_type]

        model = cls(**config).to(dtype=torch_dtype, device=device)

        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
        # avoids materializing the whole fp32 checkpoint in memory if torch dtype if fp16

        embedding_map = {
            'transformer.wte.weight':      'embedding.tok_embed.weight',
            'transformer.wpe.weight':      'embedding.pos_embed.weight'
        }

        lm_head_map = {
            'transformer.ln_f.weight':     'lm_head.ln.weight',
            'transformer.ln_f.bias':       'lm_head.ln.bias'
        }

        stack_map = lambda i: {
            f'transformer.h.{i}.ln_1.weight':           f'stack.{i}.ln1.weight',
            f'transformer.h.{i}.ln_1.bias':             f'stack.{i}.ln1.bias',
            f'transformer.h.{i}.attn.c_attn.weight':    f'stack.{i}.mhattn.qkv_w.weight',
            f'transformer.h.{i}.attn.c_attn.bias':      f'stack.{i}.mhattn.qkv_w.bias',
            f'transformer.h.{i}.attn.c_proj.weight':    f'stack.{i}.mhattn.proj_w.weight',
            f'transformer.h.{i}.attn.c_proj.bias':      f'stack.{i}.mhattn.proj_w.bias',
            f'transformer.h.{i}.ln_2.weight':           f'stack.{i}.ln2.weight',
            f'transformer.h.{i}.ln_2.bias':             f'stack.{i}.ln2.bias',
            f'transformer.h.{i}.mlp.c_fc.weight':       f'stack.{i}.mlpf.emb_ff.weight',
            f'transformer.h.{i}.mlp.c_fc.bias':         f'stack.{i}.mlpf.emb_ff.bias',
            f'transformer.h.{i}.mlp.c_proj.weight':     f'stack.{i}.mlpf.ff_emb.weight',
            f'transformer.h.{i}.mlp.c_proj.bias':       f'stack.{i}.mlpf.ff_emb.bias'
        }

        all_mappings = {**embedding_map, **lm_head_map}
        for i in range(config['num_layers']):
            all_mappings.update(stack_map(i))
        
        #conv1d checkpoints
        needs_T = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        src_params  = dict(model_hf.named_parameters())
        tgt_params  = dict(model.named_parameters())

        assert len(src_params) == len(tgt_params), "mismatch between src and tgt params"

        with torch.no_grad():
            for hf_k, our_k in all_mappings.items():
                src = src_params.get(hf_k)
                tgt = tgt_params.get(our_k)

                assert src is not None and tgt is not None, f"missing key mapping: {hf_k}->{our_k}"

                tensor = src
                if any(hf_k.endswith(sfx) for sfx in needs_T):
                    tensor = tensor.transpose(0, 1)

                tgt.copy_(tensor.to(dtype=tgt.dtype, device=tgt.device), non_blocking=True)

        del model_hf, src_params
        import gc
        gc.collect()

        return model
    