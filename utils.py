import requests
from pathlib import Path
from tqdm import tqdm
import os
import torch
from safetensors import safe_open
import gc


def download_safetensors(url, dir):
    save_path = Path(".cache") / dir
    save_path.mkdir(parents=True, exist_ok=True)
    model_file = save_path / "model.safetensors"
    HF_TOKEN = os.getenv('HF_TOKEN')
    if not model_file.exists():
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(model_file, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=f"Downloading weights from {url}"
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
                        pbar.update(len(chunk))
    return model_file

def stream_safetensors_to_meta_model(model, model_file, all_mappings, needs_T, torch_dtype, device):
    """
    for weight maps in all_mappings get tensor from model_file (src must be there) and then puts into model (dst must be there)
    """
    # stream tensors directly from the safetensors file into the model parameters (no big state dict)
    with torch.inference_mode():
        with safe_open(str(model_file), framework="pt", device="cpu") as f:
            f_keys = set(f.keys())
            step = 0
            for src_key, dst_path in all_mappings.items():
                assert src_key in f_keys, f"Missing key in checkpoint: {src_key}"

                t = f.get_tensor(src_key)  # CPU, memory-mapped

                if any(src_key.endswith(suffix) for suffix in needs_T):
                    t = t.T.contiguous()

                # Resolve destination parent and name
                obj = model
                parts = dst_path.split('.')
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                name = parts[-1]

                # Allocate REAL storage directly on target device
                real = torch.empty_like(t, device=device, dtype=torch_dtype)
                real.copy_(t, non_blocking=True)

                # Install the tensor into the module (replaces meta param)
                setattr(obj, name, torch.nn.Parameter(real, requires_grad=False))

                # Drop CPU refs ASAP
                del t

                step += 1
                if (step % 8) == 0:
                    gc.collect()

    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()

    return model

