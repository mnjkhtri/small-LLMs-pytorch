from models.pretrained import PretrainedSpec


def _llama3_spec(url, cache_dir):

    config = dict(vocab_size=128256, max_length=8192, embed_dim=2048, ff_dim=4*2048, num_heads=32, num_kv_heads=8, num_layers=16)

    embedding_map = {
        'model.embed_tokens.weight': 'embed.W_E.weight',
    }

    lm_head_map = {
        'model.norm.weight': 'lm_head.ln_f.weight'
    }

    def blocks_map(i):
        return {
            f'model.layers.{i}.input_layernorm.weight': f'blocks.{i}.ln1.weight',
            f'model.layers.{i}.self_attn.q_proj.weight': f'blocks.{i}.mhattn.q_proj.weight',
            f'model.layers.{i}.self_attn.k_proj.weight': f'blocks.{i}.mhattn.k_proj.weight',
            f'model.layers.{i}.self_attn.v_proj.weight': f'blocks.{i}.mhattn.v_proj.weight',
            f'model.layers.{i}.self_attn.o_proj.weight': f'blocks.{i}.mhattn.o_proj.weight',
            f'model.layers.{i}.post_attention_layernorm.weight': f'blocks.{i}.ln2.weight',
            f'model.layers.{i}.mlp.up_proj.weight': f'blocks.{i}.mlpf.up_proj.weight',
            f'model.layers.{i}.mlp.gate_proj.weight': f'blocks.{i}.mlpf.gate_proj.weight',
            f'model.layers.{i}.mlp.down_proj.weight': f'blocks.{i}.mlpf.down_proj.weight',
        }

    mappings = {**embedding_map, **lm_head_map}

    for i in range(config["num_layers"]):
        mappings.update(blocks_map(i))

    return PretrainedSpec(url=url, cache_dir=cache_dir, config=config, mappings=mappings)


LLAMA3_SPECS={
  "BASE":lambda:_llama3_spec("https://huggingface.co/meta-llama/llama-3.2-1b/resolve/main/model.safetensors", "llama-3.2-1b"),
  "INSTRUCT":lambda:_llama3_spec("https://huggingface.co/meta-llama/llama-3.2-1b-instruct/resolve/main/model.safetensors", "llama-3.2-1b-instruct"),
}