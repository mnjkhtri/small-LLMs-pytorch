import json
import regex as re
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT2BPE:

    def __init__(self, byte_to_unicode, unicode_to_byte, base_toks, regex_str):
        self.byte_to_unicode = byte_to_unicode
        self.unicode_to_byte = unicode_to_byte

        self.tok2id: Dict[str, int] = {}
        self.id2tok: Dict[int, str] = {}
        for ch in base_toks:
            i = len(self.tok2id)
            self.tok2id[ch] = i
            self.id2tok[i] = ch

        self.regex_pat = re.compile(regex_str) if regex_str is not None else None
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}

        self.sos_tok = None
        self.eos_tok = None
        # they should init by training process (or preloading) and be part of tok2id, 

    def _chunks(self, text: str) -> List[str]:
        """
        normal str -> chunkify.
        """
        if self.regex_pat is None:
            return [text]
        return [m.group(0) for m in self.regex_pat.finditer(text)]

    def _visible(self, text: str) -> str:
        """
        raw text -> utf-8 bytes -> visible stand-ins.
        """
        return "".join(self.byte_to_unicode[b] for b in text.encode("utf-8"))

    def train(self, text, vocab_size, verbose=False):
        """
        Uses the configured pre-splitter:
        regex_pat is None: stream mode (one big chunk; merges can cross anything)
        regex_pat: regex-chunked (merges confined within chunks; GPT-style)
        Training mutates tok2id/id2tok and bpe_ranks.
        """

        assert vocab_size >= len(
            self.tok2id
        ), "vocab_size must be >= 256 (base alphabet + any pre-added specials)."

        new_merges = vocab_size - len(self.tok2id)

        # precreate a freq dictonary of chunks so instead of iterating over all chunks we iterate on this
        chunks = self._chunks(text)
        from collections import Counter
        chunk_freq = Counter()
        for ch in chunks:
            vis = self._visible(ch)
            chunk_freq[tuple(vis)] += 1

        # iteratively pick the most frequent adjacent pair, assign next rank, update words
        for step in range(new_merges):
            pair_freq = {}
            for syms, f in chunk_freq.items(): # (tuple, frequency)
                if len(syms) < 2:
                    continue
                for i in range(len(syms) - 1):
                    p = (syms[i], syms[i + 1])
                    pair_freq[p] = pair_freq.get(p, 0) + f
            if not pair_freq:
                print(f"[train] stop: no pairs at step {step}")
                break

            best = max(pair_freq.items(), key=lambda kv: kv[1])[0]  # (a,b) to merge
            rank = len(self.bpe_ranks)
            self.bpe_ranks[best] = rank

            merged_tok = best[0] + best[1]
            
            assert merged_tok not in self.tok2id
            nid = len(self.tok2id)
            self.tok2id[merged_tok] = nid
            self.id2tok[nid] = merged_tok

            a, b = best

            def merge_once(syms):
                """Replace every (a,b) with 'a+b' inside one tuple of unicodes."""
                out: List[str] = []
                i = 0
                while i < len(syms):
                    if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                        out.append(merged_tok)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                return tuple(out)

            new_freq = {}
            for syms, f in chunk_freq.items():
                ms = merge_once(syms)
                new_freq[ms] = new_freq.get(ms, 0) + f

            chunk_freq = new_freq

            if verbose:
                print(f"merge {step+1}/{new_merges}: {best} -> '{merged_tok}' | occurence: {pair_freq[best]} | rank={rank}")

        # specials:
        nid = len(self.tok2id)
        self.tok2id['<sos>'] = nid
        self.id2tok[nid] = '<sos>'
        self.sos_tok = '<sos>'

        nid = len(self.tok2id)
        self.tok2id['<eos>'] = nid
        self.id2tok[nid] = '<eos>'
        self.eos_tok = '<eos>'

    def _bpe(self, vis_chunk: str) -> List[str]:
        """
        Apply BPE merges greedily (by rank) within ONE visible chunk.
        Returns a list of merged token strings (each 1+ stand-in chars).
        """
        syms: List[str] = list(vis_chunk)
        while True:
            ps = {(syms[i], syms[i + 1]) for i in range(len(syms) - 1)}
            if not ps:
                break
            # choose the present pair with the smallest rank number
            best = min(ps, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            # if best comes from inf then need to check if such merge is possible
            if best not in self.bpe_ranks:
                break
            a, b = best
            # merge all (a,b) occurrences in a single pass
            out: List[str] = []
            i = 0
            while i < len(syms):
                if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            syms = out
            if len(syms) == 1:
                break
        return syms

    def encode(self, text: str, add_specials=False, verbose=False) -> List[int]:
        """
        Convert text: str -> token IDs: List
        """

        ids: List[int] = []

        for i, chunk in enumerate(self._chunks(text)):
            vis = self._visible(chunk)
            toks = self._bpe(vis)
            tok_ids = [self.tok2id[tok] for tok in toks]
            if verbose:
                print(f"chunk {i}: {chunk!r} | visible: {vis} | toks: {toks} | ids: {tok_ids}")
            ids.extend(tok_ids)

        if add_specials:
            if self.sos_tok:
                ids = [self.tok2id[self.sos_tok]] + ids
            if self.eos_tok:
                ids.append(self.tok2id[self.eos_tok])

        return ids

    def decode(self, ids) -> str:
        """
        Convert token IDs: List -> text: str. Fully lossless (reverses the byte map).
        """
        s: str = "".join(self.id2tok[i] for i in ids)
        return bytes(self.unicode_to_byte[ch] for ch in s).decode("utf-8")

    @classmethod
    def gpt2(cls, pretrained=False):
        
        # nice bytes kept as their own codepoints
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]  # copy
        # remaining bytes remapped to U+0100, U+0101, ...
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        enc = {b: chr(c) for b, c in zip(bs, cs)}
        dec = {v: k for k, v in enc.items()}
        init_standins = [chr(c) for c in cs]

        init_bpe = cls(enc, dec, init_standins, (r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""))

        if not pretrained:
            return init_bpe
        
        # instead of training load the weights?
                
        from huggingface_hub import hf_hub_download
        vocab_path  = hf_hub_download(repo_id="openai-community/gpt2", filename="vocab.json")
        merges_path = hf_hub_download(repo_id="openai-community/gpt2", filename="merges.txt")

        with open(vocab_path, "r", encoding="utf-8") as f:
            enc = json.load(f)

        init_bpe.tok2id = {k: int(v) for k, v in enc.items()}
        init_bpe.id2tok = {int(v): k for k, v in enc.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            merges = [
                tuple(line.strip().split())
                for line in f
                if line and not line.startswith("#")
            ]

        init_bpe.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        init_bpe.eos_tok = '<|endoftext|>'
        return init_bpe
    

class GPT2Embedding(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim

        # [vocab_size x embed_dim]
        self.tok_embed = nn.Embedding(self.vocab_size, self.embed_dim)

        # [max_length x embed_dim]
        self.pos_embed = nn.Embedding(self.max_length, self.embed_dim)

    @property
    def weight(self): # convenience alias
        return self.tok_embed.weight

    def forward(self, x, x_pos):

        # x = [B, Tq], x_pos  = [1, Tq]

        x = self.tok_embed(x) + self.pos_embed(x_pos)
        # x = [B, Tq, De] + [1, Tq, De]
        # x = [B, Tq, De]
        # Note: Essentially a table lookup of id to dense representation. Hence a huge matrix
        return x

class GPT2MHAttention(nn.Module):
    def __init__(self, max_length, embed_dim, num_heads):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # deriv:
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = (embed_dim // num_heads)

        # embed_dim -> 3 * embed_dim
        self.qkv_w = nn.Linear(self.embed_dim, 3*self.embed_dim)

        # embed_dim -> embed_dim
        self.proj_w = nn.Linear(self.embed_dim, self.embed_dim)

    def _split_heads(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [B, H, Tq, Dh]

    def _merge_heads(self, x):
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H*Dh)  # [B, Tq, De]

    def forward(self, x, past_kv=None, use_cache=False):
        """
        x: [B, Tq, De], Tq = 1 for inference (usually)
        past_kv: (K_cache, V_cache) where K_cache, V_cache: [B, H, Tpast, Dh], essentially the neural memory;
        use_cache: if True, return (y, (K_total, V_total)) else return y;
        """

        # [B, Tq, De]

        B, T_q, _ = x.size()
        x = self.qkv_w(x)
        # [B, Tq, 3 * De]

        qx, kx, vx = x.split(self.embed_dim, dim=2) # each [B, Tq, De]

        qx, kx, vx = map(self._split_heads, (qx, kx, vx)) 
        # each [B, H, Tq, Dh]

        mask = torch.tril(torch.ones(self.max_length, self.max_length, dtype=torch.bool)).view(1, 1, self.max_length, self.max_length).to(x.device)

        if past_kv is not None:
            kp, vp = past_kv                                                # [B, H, Tpast, Dh]
            T_past = kp.size(2)
            Kx = torch.cat([kp, kx], dim=2)                                 # [B, H, Tpast + Tq, Dh]
            Vx = torch.cat([vp, vx], dim=2)                                 # [B, H, Tpast + Tq, Dh]
            attn_mask = mask[:, :, T_past:T_past+T_q, :T_past+T_q]          # [1, 1, Tq, Tpast + Tq]
        else:
            Kx, Vx = kx, vx
            T_past = 0
            attn_mask = mask[:, :, :T_q, :T_q]                              # [1, 1, Tq, Tq]

        att_w = torch.einsum('bhqd,bhkd->bhqk', [qx, Kx]) / (self.head_dim ** 0.5)
        # [B, H, Tq, Tpast + Tq]
        att_w = att_w.masked_fill(~attn_mask, torch.finfo(att_w.dtype).min)
        # [B, H, Tq, Tpast + Tq]

        att_w = F.softmax(att_w, dim=-1)
        # [B, H, Tq, Tpast + Tq]

        out = torch.einsum('bhal,bhlv->bhav', [att_w, Vx])
        # [B, H, Tq, Dh]

        y = self.proj_w(self._merge_heads(out))
        # y = [B, Tq, De]

        return (y, (Kx, Vx)) if use_cache else y

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

class GPT2MLPF(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim

        self.emb_ff = nn.Linear(self.embed_dim, self.ff_dim)
        self.ff_emb = nn.Linear(self.ff_dim, self.embed_dim)

    def forward(self, x):

        # [B, Tq, De]

        x = self.emb_ff(x)
        # [B, Tq, Df]

        x = F.gelu(x, approximate="tanh")
        x = self.ff_emb(x)
        # [B, Tq, De]

        return x
    
        # [B, Tq, De]
    
        # Note: The affine transformaiton E -> E is not very expessive. By expanding four times and compressive back learns rich features.
        # This acts like a bottleneck autoencoder
        # In a stack 2/3rd of parameters live in MLPF while rest 1/3rd is in MHA

class GPT2Block(nn.Module):
    def __init__(self, max_length, embed_dim, ff_dim, num_heads):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads

        self.mhattn = GPT2MHAttention(self.max_length, self.embed_dim, self.num_heads)
        self.mlpf = GPT2MLPF(self.embed_dim, self.ff_dim)
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x, past_kv=None, use_cache=False):

        # [B, Tq, De]

        if use_cache:
            attd_x, present_kv = self.mhattn(self.ln1(x), past_kv=past_kv, use_cache=True)
            x = x + attd_x
            x = x + self.mlpf(self.ln2(x))
            return x, present_kv
        else:
            x = x + self.mhattn(self.ln1(x))
            x = x + self.mlpf(self.ln2(x))
            return x

        # [B, Tq, De]

class GPT2LMHead(nn.Module):
    def __init__(self, embed_dim, embed_wt):
        super().__init__()
        self.embed_dim = embed_dim

        self.ln = nn.LayerNorm(self.embed_dim)
        self.embed_wt = embed_wt

    def forward(self, x):

        # [B, Tq, De]

        w = self.embed_wt.weight
        x = self.ln(x)
        out = F.linear(x, w)
        # [B, Tq, De]

        return out
        
class GPT2(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim, ff_dim, num_heads, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embed = GPT2Embedding(self.vocab_size, self.max_length, self.embed_dim)
        self.lm_head = GPT2LMHead(self.embed_dim, self.embed)

        self.blocks = nn.ModuleList(
            [GPT2Block(self.max_length, self.embed_dim, self.ff_dim, self.num_heads) for _ in range(self.num_layers)]
        )

    def forward(self, x, past_key_values=None, use_cache=False):

        B, T = x.size()

        if past_key_values is not None and past_key_values[0] is not None:
            T_past = past_key_values[0][0].size(2)
        else:
            T_past = 0

        assert (T + T_past) <= self.max_length, "sequence length cant exceed max length"

        # [B, Tq]

        x_pos = torch.arange(T_past, T_past + T, device=x.device).unsqueeze(0)  # [1, T]
        x = self.embed(x, x_pos=x_pos)
        # [B, Tq, De]

        if use_cache:

            if past_key_values is None: past_key_values = [None] * self.num_layers
            # 1st inference init

            new_past_key_values = []

            for i, block in enumerate(self.blocks):
                x, present_kv = block(x, past_kv=past_key_values[i], use_cache=True)
                new_past_key_values.append(present_kv)
            # [B, Tq, De]

            x = self.lm_head(x)
            # [B, Tq, VOCAB_SIZE]

            return {
                'logits': x,
                'past_key_values': new_past_key_values
            }

        else:
            for block in self.blocks:
                x = block(x)
            # [B, Tq, De]

            x = self.lm_head(x)
            # [B, Tq, VOCAB_SIZE]

            return {
                'logits': x,
            }
    
    @classmethod
    def from_pretrained(cls, model_type, *, torch_dtype=torch.float32, device="cuda"):

        assert model_type in ('BASE'), "only supports BASE"
        
        config = dict(vocab_size=50257, max_length=1024, embed_dim=768,  ff_dim=768*4,  num_heads=12, num_layers=12)
        model = cls(**config).to(dtype=torch_dtype, device=device)

        from utils import download_safetensors, stream_safetensors_to_meta_model

        MODEL_URL = 'https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors'
        DIR = 'gpt2'

        model_file = download_safetensors(MODEL_URL, DIR)

        with torch.device('meta'):
            model = cls(**config)

        embedding_map = {
            'wte.weight':      'embed.tok_embed.weight',
            'wpe.weight':      'embed.pos_embed.weight'
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

        needs_T = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        model = stream_safetensors_to_meta_model(model, model_file, all_mappings, needs_T, torch_dtype, device)

        tokenizer = GPT2BPE.gpt2(pretrained=True)

        return tokenizer, model
    