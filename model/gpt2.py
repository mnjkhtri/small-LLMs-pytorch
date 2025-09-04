from tokenizer.gpt2 import BPE
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
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = (embed_dim // num_heads)

        # fused qkv projection:
        self.qkv_w = nn.Linear(self.embed_dim, 3*self.embed_dim)
        self.proj_w = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.qkv_w(x)
        qx, kx, vx = x.split(self.embed_dim, dim=2)
        qx = qx.view(x.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kx = kx.view(x.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        vx = vx.view(x.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        att_w = torch.einsum('bhqd,bhkd->bhqk', [qx, kx])/(self.head_dim ** 0.5)
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
        x = 0.5 * x * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)))
        x = self.ff_emb(x)
        x = self.dropout(x)
        return x

class GPT2Block(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads

        self.mhattn = MHAttention(embed_dim, num_heads, dropout)
        self.mlpf = MLPF(embed_dim, ff_dim, dropout)

        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        x = x + self.mhattn(self.ln1(x))
        x = x + self.mlpf(self.ln2(x))
        return x

class GPT2LMHead(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        # [B, SEQ_LENGTH, EMBED_DIM]
        return self.lm_head(x)
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

        self.stack = nn.Sequential(*[GPT2Block(embed_dim, ff_dim, num_heads, dropout) for _ in range(self.num_layers)])

        self.lm_head = GPT2LMHead(self.embed_dim, self.vocab_size)
        
    def forward(self, x):

        # [B, SEQ_LENGTH]
        x = self.embedding(x)
        # [B, SEQ_LENGTH, EMBED_DIM]
        x = self.stack(x)
        # [B, SEQ_LENGTH, EMBED_DIM]
        x = self.lm_head(x)

        return x

# class GPT2:

#     def __init__(self):

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         model = 'gpt2'
#         self.tokenizer = BPE.gpt2_bpe(pretrained=True)
#         self.model = AutoModelForCausalLM.from_pretrained(model).to(self.device)
#         self.model.eval()

#         print(self.model.state_dict().keys())

#     def _infer_once(self, 
#                     prompt: str,
#                     *,
#                     top_k: int = 1,
#                     T: float = 1.0
#                 ) -> str:

#         prompt_ids = self.tokenizer.encode(prompt)
#         input_ids = torch.tensor([prompt_ids], device=self.device)
#         # int64: [1, SEQ]

#         with torch.no_grad(): 
#             out = self.model(input_ids=input_ids, use_cache=False)
#             # [1, SEQ, VOCAB]
#             next_logits = out.logits[:, -1, :] 
#             # [1, VOCAB]

#         # Handle TopK (replace < cutoff with -inf so they are softmaxx to 0):
#         topk_logits, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1])) # [1, TOPK]
#         cutoff = topk_logits[..., -1].unsqueeze(-1) # [1, TOPK]
#         next_logits = torch.where(next_logits < cutoff, torch.full_like(next_logits, -float("inf")), next_logits)

#         # Sampling with Temp:
#         probs = torch.softmax(next_logits / T, dim=-1)
#         next_id = torch.multinomial(probs, num_samples=1) 
#         # [1,1]

#         return self.tokenizer.decode([next_id[0].item()])
    
#     def generate(self, 
#                 prompt: str,
#                 *,
#                 top_k: int = 1,
#                 T: float = 1.0
#             ) -> str:

#         new_prompt = prompt
#         while True:
#             new_tok = self._infer_once(
#                 new_prompt,
#                 top_k=top_k,
#                 T=T
#             )
#             new_prompt += new_tok
#             print(new_tok, end="", flush=True)

if __name__ == "__main__":

    tokenizer = BPE.gpt2_bpe(pretrained=True)

    vocab_size = 50257
    max_length = 1024
    embed_dim=768
    ff_dim=768*4
    num_heads=12
    layers=12
    dropout = 0.1

    gpt2 = GPT2(vocab_size, max_length, embed_dim, ff_dim, num_heads, layers, dropout=0.1).to('cuda')

    prompt = "Life is beautiful"
    input_ids = torch.tensor(tokenizer.encode(prompt)).view(1, -1).to('cuda')

    out = gpt2.forward(input_ids)
    print(out.shape)

    # print(prompt, end='', flush=True)
    # gpt2.generate(prompt, top_k=100, T=1.0)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # prompt = "The answer to life, universe and everything is"
    # model_type = 'gpt2'
    
    # model = Transformer.from_pretrained(model_type)
    # model.to(device)
    # model.eval()
    # import tiktoken
    # tokenizer = tiktoken.get_encoding("gpt2")
    # indices = torch.tensor(tokenizer.encode(prompt)).view(1, -1).to(device)
    # out = model.generate(indices, 100, do_sample=True)
    # print(tokenizer.decode(out.detach().tolist()[0]))
