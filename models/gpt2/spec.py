from models.pretrained import PretrainedSpec


def gpt2_spec():

    config = dict(vocab_size=50257, max_length=1024, d_model=768, d_mlp=768*4, n_heads=12, n_layers=12)

    embedding_map = {
        "wte.weight": "embed.W_E.weight",
        "wpe.weight": "embed.W_pos.weight"
    }

    lm_head_map = {
        "ln_f.weight": "lm_head.ln_f.weight",
        "ln_f.bias": "lm_head.ln_f.bias"
    }

    def blocks_map(i):
        return {
            f"h.{i}.ln_1.weight": f"blocks.{i}.ln1.weight",
            f"h.{i}.ln_1.bias": f"blocks.{i}.ln1.bias",
            f"h.{i}.attn.c_attn.weight": f"blocks.{i}.mhattn.W_QKV.weight",
            f"h.{i}.attn.c_attn.bias": f"blocks.{i}.mhattn.W_QKV.bias",
            f"h.{i}.attn.c_proj.weight": f"blocks.{i}.mhattn.W_O.weight",
            f"h.{i}.attn.c_proj.bias": f"blocks.{i}.mhattn.W_O.bias",
            f"h.{i}.ln_2.weight": f"blocks.{i}.ln2.weight",
            f"h.{i}.ln_2.bias": f"blocks.{i}.ln2.bias",
            f"h.{i}.mlp.c_fc.weight": f"blocks.{i}.mlpf.W_in.weight",
            f"h.{i}.mlp.c_fc.bias": f"blocks.{i}.mlpf.W_in.bias",
            f"h.{i}.mlp.c_proj.weight": f"blocks.{i}.mlpf.W_out.weight",
            f"h.{i}.mlp.c_proj.bias": f"blocks.{i}.mlpf.W_out.bias",
        }

    mappings = {**embedding_map, **lm_head_map}
    
    for i in range(config["n_layers"]):
        mappings.update(blocks_map(i))

    needs_T = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

    return PretrainedSpec(
        url="https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors",
        cache_dir="gpt2",
        config=config,
        mappings=mappings,
        needs_T=needs_T
    )

GPT2_SPECS = {"BASE": gpt2_spec}
