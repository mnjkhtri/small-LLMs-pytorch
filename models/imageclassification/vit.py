import torch
import torch.nn as nn
import torch.nn.functional as F

import io
import requests
from PIL import Image
from torchvision import transforms


class ViTPreprocessor:
    def __init__(self):
        pass

    def from_url(self, image_url):

        response = requests.get(image_url)
        img = Image.open(io.BytesIO(response.content)).convert('RGB')

        preprocess = transforms.Compose([
            transforms.Resize(256),             
            transforms.CenterCrop(224),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img = preprocess(img).unsqueeze(0)
        return img
    

class ViTEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.cls_tok = nn.Parameter(torch.empty(1, 1, self.embed_dim))
        self.pos_embeddings = nn.Parameter(torch.empty(1, self.num_patches + 1, self.embed_dim))
        self.pat_proj = nn.Conv2d(in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        #(B, 3, IMAGE_SIZE, IMAGE_SIZE)
        B = x.shape[0]

        # image is divided into patches and each patch (3 channels depth) is converted into one pixel (De channels deep)

        x = self.pat_proj(x)
        #(B, De, IMAGE_SIZE // PATCH_SIZE, IMAGE_SIZE // PATCH_SIZE)

        x = x.view(x.shape[0], self.embed_dim, self.num_patches).permute(0, 2, 1)
        #(B, T, De)

        cls_tokens = self.cls_tok.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        #(B, T+1, De)
        
        x = x + self.pos_embeddings
        #(B, T+1, De)

        return self.dropout(x)

class ViTLMHead(nn.Module):
    def __init__(self, embed_dim, no_classes):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, no_classes)

    def forward(self, x):

        # [B, T+1, De]

        out = self.classifier(x[:, 0, :])
        # [B, NO_CLASSES]

        return out
        
class ViTMHAttention(nn.Module):
    def __init__(self, num_patches, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.max_length = num_patches
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = (embed_dim // num_heads)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.dropout = nn.Dropout(0.1)

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
        att_w = self.dropout(att_w) # still [B, H, T, T]

        out = torch.einsum('bhal,bhlv->bhav', [att_w, vx])
        # [B, H, T, Dh]

        out = out.permute(0,2,1,3).contiguous()
        # [B, T, H, Dh]

        out = out.view(B, -1, self.num_heads * self.head_dim)
        # [B, T, De]

        out = self.dropout(self.o_proj(out)) 
        # [B, T, De]

        return out

class ViTMLPF(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim

        self.emb_ff = nn.Linear(self.embed_dim, self.ff_dim)
        self.ff_emb = nn.Linear(self.ff_dim, self.embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        #[B, T+1, De]
        x = self.emb_ff(x)
        x = F.gelu(x, approximate="tanh")
        #[B, T+1, Df]

        x = self.ff_emb(x)
        #[B, T+1, De]

        x = self.dropout(x)
        #[B, T+1, De]

        return x

class ViTBlock(nn.Module):
    def __init__(self, num_patches, embed_dim, ff_dim, num_heads):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads

        self.mhattn = ViTMHAttention(self.num_patches, self.embed_dim, self.num_heads)
        self.mlpf = ViTMLPF(self.embed_dim, self.ff_dim)

        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        
        # [B, E, De]

        x = x + self.mhattn(self.ln1(x))
        # [B, E, De]

        x = x + self.mlpf(self.ln2(x))
        # [B, E, De]

        return x

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, ff_dim, num_heads, num_layers, no_classes):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.no_classes = no_classes

        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.embed = ViTEmbedding(self.image_size, self.patch_size, self.embed_dim)
        self.lm_head = ViTLMHead(self.embed_dim, self.no_classes)

        self.blocks = nn.ModuleList(
            [ViTBlock(self.num_patches, self.embed_dim, self.ff_dim, self.num_heads) for _ in range(self.num_layers)]
        )

    def forward(self, x):
        # [B, 3, IMAGE_SIZE, IMAGE_SIZE]

        x = self.embed(x)
        # [B, T+1, De]

        for block in self.blocks:
            x = block(x)

        # [B, T+1, De]

        x = self.lm_head(x)
        # [B, NO_CLASSES]

        return {
            'logits': x,
        }
    
    @classmethod
    def from_pretrained(cls, *, torch_dtype=torch.float32, device="cuda"):

        config = dict(image_size=224, patch_size=16, embed_dim=768,  ff_dim=768*4,  num_heads=12, num_layers=12, no_classes=1000)
        model = cls(**config).to(dtype=torch_dtype, device=device)

        from utils import download_safetensors, stream_safetensors_to_meta_model

        MODEL_URL = 'https://huggingface.co/google/vit-base-patch16-224/resolve/main/model.safetensors'
        DIR = 'vit'

        # 1. model file
        model_file = download_safetensors(MODEL_URL, DIR)

        # 2. meta nn module
        with torch.device('meta'):
            model = cls(**config)

        # 3. embeddings
        embedding_map = {
            'vit.embeddings.position_embeddings':                   'embedding.pos_embeddings',
            'vit.embeddings.patch_embeddings.projection.bias':      'embedding.pat_proj.bias',
            'vit.embeddings.patch_embeddings.projection.weight':    'embedding.pat_proj.weight',
            'vit.embeddings.cls_token':                             'embedding.cls_tok',
        }

        lm_head_map = {
            'classifier.weight':     'lm_head.classifier.weight',
            'classifier.bias':       'lm_head.classifier.bias'
        }

        blocks_map = lambda i: {
            f'vit.encoder.layer.{i}.attention.attention.query.weight':  f'blocks.{i}.mhattn.q_proj.weight',
            f'vit.encoder.layer.{i}.attention.attention.query.bias':    f'blocks.{i}.mhattn.q_proj.bias',
            f'vit.encoder.layer.{i}.attention.attention.key.weight':    f'blocks.{i}.mhattn.k_proj.weight',
            f'vit.encoder.layer.{i}.attention.attention.key.bias':      f'blocks.{i}.mhattn.k_proj.bias',
            f'vit.encoder.layer.{i}.attention.attention.value.weight':  f'blocks.{i}.mhattn.v_proj.weight',
            f'vit.encoder.layer.{i}.attention.attention.value.bias':    f'blocks.{i}.mhattn.v_proj.bias',
            f'vit.encoder.layer.{i}.attention.output.dense.weight':     f'blocks.{i}.mhattn.o_proj.weight',
            f'vit.encoder.layer.{i}.attention.output.dense.bias':       f'blocks.{i}.mhattn.o_proj.bias',
            f'vit.encoder.layer.{i}.intermediate.dense.weight':         f'blocks.{i}.mlpf.emb_ff.weight',
            f'vit.encoder.layer.{i}.intermediate.dense.bias':           f'blocks.{i}.mlpf.emb_ff.bias',
            f'vit.encoder.layer.{i}.output.dense.weight':               f'blocks.{i}.mlpf.ff_emb.weight',
            f'vit.encoder.layer.{i}.output.dense.bias':                 f'blocks.{i}.mlpf.ff_emb.bias',
            f'vit.encoder.layer.{i}.layernorm_before.weight':           f'blocks.{i}.ln1.weight',
            f'vit.encoder.layer.{i}.layernorm_before.bias':             f'blocks.{i}.ln1.bias',
            f'vit.encoder.layer.{i}.layernorm_after.weight':            f'blocks.{i}.ln2.weight',
            f'vit.encoder.layer.{i}.layernorm_after.bias':              f'blocks.{i}.ln2.bias'
        }

        all_mappings = {**embedding_map, **lm_head_map}
        for i in range(config['num_layers']):
            all_mappings.update(blocks_map(i))

        model = stream_safetensors_to_meta_model(model, model_file, all_mappings, [], torch_dtype, device)
        processor = ViTPreprocessor()
        return processor, model
