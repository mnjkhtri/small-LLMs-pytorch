import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from pathlib import Path
import io
import os
import requests
from PIL import Image
from torchvision import transforms


class CLIPPreprocessor:
    def __init__(self):
        pass

    def from_url(self, image_url):
        response = requests.get(image_url)
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(256),             
            transforms.CenterCrop(224),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        img = preprocess(img).unsqueeze(0)
        return img

class CLIPVisionEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, hidden_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.cls_tok = nn.Parameter(torch.empty(self.hidden_dim))
        self.register_buffer(
            "position_ids", 
            torch.arange(self.num_patches+1).expand((1, -1))
        )
        self.pos_embed = nn.Embedding(self.num_patches+1, self.hidden_dim)
        self.pat_proj = nn.Conv2d(in_channels=3, out_channels=self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)

    def forward(self, x):

        #(B, 3, IMAGE_SIZE, IMAGE_SIZE)
        B = x.shape[0]

        # image is divided into patches and each patch (3 channels depth) is converted into one pixel (De channels deep)

        x = self.pat_proj(x)
        #(B, De, IMAGE_SIZE // PATCH_SIZE, IMAGE_SIZE // PATCH_SIZE)

        x = x.view(x.shape[0], self.hidden_dim, self.num_patches).permute(0, 2, 1)
        #(B, T, De)

        cls_tokens = self.cls_tok.expand(B, 1, self.hidden_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        #(B, T+1, De)
        
        x = x + self.pos_embed(self.position_ids)
        #(B, T+1, De)

        return x

class CLIPVisionMHAttention(nn.Module):
    def __init__(self, num_patches, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.max_length = num_patches
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = (hidden_dim // num_heads)

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

    def forward(self, x):

        # [B, T, De]

        B, T, _ = x.size()

        qx = self.q_proj(x)
        kx = self.k_proj(x)
        vx = self.v_proj(x) # each [B, T, De]

        qx = qx.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kx = kx.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        vx = vx.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # each [B, H, T, Dh]

        att_w = torch.einsum('bhqd,bhkd->bhqk', [qx, kx]) / (self.head_dim ** 0.5) #  [B, H, T, T]
        att_w = F.softmax(att_w, dim=-1)

        out = torch.einsum('bhal,bhlv->bhav', [att_w, vx])
        # [B, H, T, Dh]

        out = out.permute(0,2,1,3).contiguous()
        # [B, T, H, Dh]

        out = out.view(B, -1, self.num_heads * self.head_dim)
        # [B, T, De]

        out = self.o_proj(out)
        # [B, T, De]

        return out

class CLIPVisionMLPF(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        self.emb_ff = nn.Linear(self.hidden_dim, self.intermediate_dim)
        self.ff_emb = nn.Linear(self.intermediate_dim, self.hidden_dim)

    def forward(self, x):

        #[B, T+1, De]
        x = self.emb_ff(x)
        x = F.gelu(x, approximate="tanh")
        #[B, T+1, Df]

        x = self.ff_emb(x)
        #[B, T+1, De]

        return x

class CLIPVisionBlock(nn.Module):
    def __init__(self, num_patches, hidden_dim, intermediate_dim, num_heads):
        super().__init__()
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads

        self.mhattn = CLIPVisionMHAttention(self.num_patches, self.hidden_dim, self.num_heads)
        self.mlpf = CLIPVisionMLPF(self.hidden_dim, self.intermediate_dim)

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, x):
        
        # [B, E, De]

        x = x + self.mhattn(self.ln1(x))
        # [B, E, De]

        x = x + self.mlpf(self.ln2(x))
        # [B, E, De]

        return x

class CLIPVision(nn.Module):
    def __init__(self, image_size, patch_size, hidden_dim, intermediate_dim, num_heads, num_layers):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.embed = CLIPVisionEmbedding(self.image_size, self.patch_size, self.hidden_dim)

        self.blocks = nn.ModuleList(
            [CLIPVisionBlock(self.num_patches, self.hidden_dim, self.intermediate_dim, self.num_heads) for _ in range(self.num_layers)]
        )

        self.pre_ln = nn.LayerNorm(self.hidden_dim)
        self.post_ln = nn.LayerNorm(self.hidden_dim)

    def forward(self, x):
        # [B, 3, IMAGE_SIZE, IMAGE_SIZE]

        x = self.embed(x)
        # [B, T+1, De]

        x = self.pre_ln(x)
        # [B, T+1, De]

        for block in self.blocks:
            x = block(x)

        # [B, T+1, De]
        
        x = self.post_ln(x)
        # [B, T+1, De]

        x = self.pre_ln(x)

        return {
            'logits': x,
        }

class CLIPHFTokenizer:
    def __init__(self, model_path):
        self.tokenizer = Tokenizer.from_file(model_path)
    
    def encode(self, text: str):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def encode_texts(self, texts):
        self.tokenizer.enable_padding()
        return [te.ids for te in self.tokenizer.encode_batch(texts)]
    
    @classmethod
    def clip(cls):
        TOKENIZER_URL = "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer.json"
        save_path = Path(".cache") / "clip"
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
    
class CLIPTextEmbedding(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_dim = hidden_dim

        self.tok_embed = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.register_buffer(
            "position_ids", 
            torch.arange(self.max_length).expand((1, -1))
        )
        self.pos_embed = nn.Embedding(self.max_length, self.hidden_dim)

    @property
    def weight(self): # convenience alias
        return self.tok_embed.weight

    def forward(self, x):

        Tq = x.shape[1]
        # [B, Tq]

        x = self.tok_embed(x) + self.pos_embed(self.position_ids[:, :Tq])
        # [B, Tq, De] + [1, Tq, De]
        # [B, Tq, De]

        return x
        
class CLIPTextMHAttention(nn.Module):
    def __init__(self, max_length, hidden_dim, num_heads):
        super().__init__()
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = (hidden_dim // num_heads)

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

    def forward(self, x):
        # [B, T, De]

        B, T, _ = x.size()

        qx = self.q_proj(x)
        kx = self.k_proj(x)
        vx = self.v_proj(x) # each [B, T, De]

        qx = qx.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kx = kx.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        vx = vx.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # each [B, H, T, Dh]

        mask = torch.tril(torch.ones(self.max_length, self.max_length, dtype=torch.bool)).view(1, 1, self.max_length, self.max_length).to(x.device)

        attn_mask = mask[:, :, :T, :T] # [1, 1, T, T]

        att_w = torch.einsum('bhqd,bhkd->bhqk', [qx, kx]) / (self.head_dim ** 0.5)
        # [B, H, T, T]
        att_w = att_w.masked_fill(~attn_mask, torch.finfo(att_w.dtype).min)
        # [B, H, T, T]

        att_w = F.softmax(att_w, dim=-1)
        # [B, H, T, T]

        out = torch.einsum('bhal,bhlv->bhav', [att_w, vx])
        # [B, H, T, Dh]

        out = out.permute(0, 2, 1, 3).contiguous()
        # [B, T, H, Dh]

        out = out.view(B, -1, self.num_heads * self.head_dim)
        # [B, T, De]

        out = self.o_proj(out)
        
        return out
    
class CLIPTextMLPF(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        self.emb_ff = nn.Linear(self.hidden_dim, self.intermediate_dim)
        self.ff_emb = nn.Linear(self.intermediate_dim, self.hidden_dim)

    def forward(self, x):

        # [B, Tq, De]

        x = self.emb_ff(x)
        # [B, Tq, Df]

        x = F.gelu(x, approximate="tanh")
        x = self.ff_emb(x)
        # [B, Tq, De]

        return x

class CLIPTextBlock(nn.Module):
    def __init__(self, max_length, hidden_dim, intermediate_dim, num_heads):
        super().__init__()
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads

        self.mhattn = CLIPTextMHAttention(self.max_length, self.hidden_dim, self.num_heads)
        self.mlpf = CLIPTextMLPF(self.hidden_dim, self.intermediate_dim)
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, x):

        # [B, T, De]

        x = x + self.mhattn(self.ln1(x))
        x = x + self.mlpf(self.ln2(x))
        return x

        # [B, T, De]

class CLIPText(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_dim, intermediate_dim, num_heads, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embed = CLIPTextEmbedding(self.vocab_size, self.max_length, self.hidden_dim)

        self.blocks = nn.ModuleList(
            [CLIPTextBlock(self.max_length, self.hidden_dim, self.intermediate_dim, self.num_heads) for _ in range(self.num_layers)]
        )

        self.final_ln = nn.LayerNorm(self.hidden_dim)

    def forward(self, x):
        
        B, T = x.size()
        assert T <= self.max_length, "sequence length cant exceed max length"

        # [B, T]

        x = self.embed(x)
        # [B, T, De]

        for block in self.blocks:
            x = block(x)
        # [B, T, De]

        x = self.final_ln(x)
        # [B, T, De]

        return {
            'logits': x,
        }


class CLIP(nn.Module):
    def __init__(self, vision_config, text_config, proj_dim):
        super().__init__()

        self.vision = CLIPVision(
            image_size=vision_config['image_size'],
            patch_size=vision_config['patch_size'],
            hidden_dim=vision_config['hidden_dim'],
            intermediate_dim=vision_config['intermediate_dim'],
            num_heads=vision_config['num_heads'],
            num_layers=vision_config['num_layers']
        )

        self.text = CLIPText(
            vocab_size=text_config['vocab_size'],
            max_length=text_config['max_length'],
            hidden_dim=text_config['hidden_dim'],
            intermediate_dim=text_config['intermediate_dim'],
            num_heads=text_config['num_heads'],
            num_layers=text_config['num_layers']
        )
        self.vision_proj = nn.Linear(vision_config['hidden_dim'], proj_dim, bias=False)
        self.text_proj = nn.Linear(text_config['hidden_dim'], proj_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, img, txt_ids):

        # Vision: Extract CLS token
        vis_out = self.vision(img)["logits"]
        cls_tok = vis_out[:, 0]
        img_emb = self.vision_proj(cls_tok)

        # Text: Extract EOS token (argmax finds max ID, typically EOS in CLIP)
        txt_out = self.text(txt_ids)["logits"]
        eos_idx = txt_ids.argmax(dim=-1)
        batch   = torch.arange(len(txt_ids), device=txt_ids.device)
        eos_tok = txt_out[batch, eos_idx]
        txt_emb = self.text_proj(eos_tok)

        # Normalize:
        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)

        return img_emb, txt_emb

    @classmethod
    def from_pretrained(cls, *, torch_dtype=torch.bfloat16, device="cuda"):

        vision_config = dict(
            image_size=224, patch_size=14, hidden_dim=1024, intermediate_dim=4096,  num_heads=16, num_layers=24,
        )
        text_config = dict(
            vocab_size=49408, max_length=77, hidden_dim=768, intermediate_dim=3072, num_heads=12, num_layers=12,
        )
        config = dict(
            vision_config=vision_config,
            text_config=text_config,
            proj_dim=768
        )

        from utils import download_safetensors, stream_safetensors_to_meta_model

        MODEL_URL = 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors'
        DIR = 'clip'

        # 1. model file
        model_file = download_safetensors(MODEL_URL, DIR)

        # 2. meta nn module
        with torch.device('meta'):
            model = cls(**config)

        # 3. embeddings
        vision_map = {
            'vision_model.embeddings.class_embedding':                       'vision.embed.cls_tok',
            'vision_model.embeddings.position_ids':                          'vision.embed.position_ids',
            'vision_model.embeddings.position_embedding.weight':             'vision.embed.pos_embed.weight',
            'vision_model.embeddings.patch_embedding.weight':                'vision.embed.pat_proj.weight',
            'vision_model.pre_layrnorm.weight':                              'vision.pre_ln.weight',
            'vision_model.pre_layrnorm.bias':                                'vision.pre_ln.bias',
            'vision_model.post_layernorm.weight':                            'vision.post_ln.weight',
            'vision_model.post_layernorm.bias':                              'vision.post_ln.bias',
        }

        vision_blocks_map = lambda i: {
            f'vision_model.encoder.layers.{i}.self_attn.q_proj.weight':      f'vision.blocks.{i}.mhattn.q_proj.weight',
            f'vision_model.encoder.layers.{i}.self_attn.q_proj.bias':        f'vision.blocks.{i}.mhattn.q_proj.bias',
            f'vision_model.encoder.layers.{i}.self_attn.k_proj.weight':      f'vision.blocks.{i}.mhattn.k_proj.weight',
            f'vision_model.encoder.layers.{i}.self_attn.k_proj.bias':        f'vision.blocks.{i}.mhattn.k_proj.bias',
            f'vision_model.encoder.layers.{i}.self_attn.v_proj.weight':      f'vision.blocks.{i}.mhattn.v_proj.weight',
            f'vision_model.encoder.layers.{i}.self_attn.v_proj.bias':        f'vision.blocks.{i}.mhattn.v_proj.bias',
            f'vision_model.encoder.layers.{i}.self_attn.out_proj.weight':    f'vision.blocks.{i}.mhattn.o_proj.weight',
            f'vision_model.encoder.layers.{i}.self_attn.out_proj.bias':      f'vision.blocks.{i}.mhattn.o_proj.bias',
            f'vision_model.encoder.layers.{i}.mlp.fc1.weight':               f'vision.blocks.{i}.mlpf.emb_ff.weight',
            f'vision_model.encoder.layers.{i}.mlp.fc1.bias':                 f'vision.blocks.{i}.mlpf.emb_ff.bias',
            f'vision_model.encoder.layers.{i}.mlp.fc2.weight':               f'vision.blocks.{i}.mlpf.ff_emb.weight',
            f'vision_model.encoder.layers.{i}.mlp.fc2.bias':                 f'vision.blocks.{i}.mlpf.ff_emb.bias',
            f'vision_model.encoder.layers.{i}.layer_norm1.weight':           f'vision.blocks.{i}.ln1.weight',
            f'vision_model.encoder.layers.{i}.layer_norm1.bias':             f'vision.blocks.{i}.ln1.bias',
            f'vision_model.encoder.layers.{i}.layer_norm2.weight':           f'vision.blocks.{i}.ln2.weight',
            f'vision_model.encoder.layers.{i}.layer_norm2.bias':             f'vision.blocks.{i}.ln2.bias'
        }

        text_map = {
            'text_model.embeddings.token_embedding.weight':         'text.embed.tok_embed.weight',
            'text_model.embeddings.position_embedding.weight':      'text.embed.pos_embed.weight',
            'text_model.final_layer_norm.weight':                   'text.final_ln.weight',
            'text_model.final_layer_norm.bias':                     'text.final_ln.bias',
            'text_model.embeddings.position_ids':                   'text.embed.position_ids'
        }

        text_blocks_map = lambda i: {
            f'text_model.encoder.layers.{i}.self_attn.q_proj.weight':      f'text.blocks.{i}.mhattn.q_proj.weight',
            f'text_model.encoder.layers.{i}.self_attn.q_proj.bias':        f'text.blocks.{i}.mhattn.q_proj.bias',
            f'text_model.encoder.layers.{i}.self_attn.k_proj.weight':      f'text.blocks.{i}.mhattn.k_proj.weight',
            f'text_model.encoder.layers.{i}.self_attn.k_proj.bias':        f'text.blocks.{i}.mhattn.k_proj.bias',
            f'text_model.encoder.layers.{i}.self_attn.v_proj.weight':      f'text.blocks.{i}.mhattn.v_proj.weight',
            f'text_model.encoder.layers.{i}.self_attn.v_proj.bias':        f'text.blocks.{i}.mhattn.v_proj.bias',
            f'text_model.encoder.layers.{i}.self_attn.out_proj.weight':    f'text.blocks.{i}.mhattn.o_proj.weight',
            f'text_model.encoder.layers.{i}.self_attn.out_proj.bias':      f'text.blocks.{i}.mhattn.o_proj.bias',
            f'text_model.encoder.layers.{i}.mlp.fc1.weight':               f'text.blocks.{i}.mlpf.emb_ff.weight',
            f'text_model.encoder.layers.{i}.mlp.fc1.bias':                 f'text.blocks.{i}.mlpf.emb_ff.bias',
            f'text_model.encoder.layers.{i}.mlp.fc2.weight':               f'text.blocks.{i}.mlpf.ff_emb.weight',
            f'text_model.encoder.layers.{i}.mlp.fc2.bias':                 f'text.blocks.{i}.mlpf.ff_emb.bias',
            f'text_model.encoder.layers.{i}.layer_norm1.weight':           f'text.blocks.{i}.ln1.weight',
            f'text_model.encoder.layers.{i}.layer_norm1.bias':             f'text.blocks.{i}.ln1.bias',
            f'text_model.encoder.layers.{i}.layer_norm2.weight':           f'text.blocks.{i}.ln2.weight',
            f'text_model.encoder.layers.{i}.layer_norm2.bias':             f'text.blocks.{i}.ln2.bias'
        }

        proj_map = {
            'logit_scale':                      'logit_scale',
            'text_projection.weight':           'text_proj.weight',
            'visual_projection.weight':         'vision_proj.weight'
        }

        all_mappings = {**proj_map, **vision_map, **text_map}

        for i in range(vision_config['num_layers']):
            all_mappings.update(vision_blocks_map(i))

        for i in range(text_config['num_layers']):
            all_mappings.update(text_blocks_map(i))

        model = stream_safetensors_to_meta_model(model, model_file, all_mappings, [], torch_dtype, device)
        return model
