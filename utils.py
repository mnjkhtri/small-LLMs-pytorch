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
    mappings_keys, mappings_values = set(all_mappings.keys()), set(all_mappings.values())
    model_keys = set(model.state_dict().keys())
    mappings_vs_model = mappings_values - model_keys
    model_vs_mappings = model_keys - mappings_values

    if mappings_vs_model:
        print(f"\n[ACTION NEEDED] The following {len(mappings_vs_model)} keys are in the MAPPINGS but NOT in your own MODEL:")
        for k in sorted(list(mappings_vs_model)):
            print(f"  + {k}")

    if model_vs_mappings:
        print(f"\n[ACTION NEEDED] The following {len(model_vs_mappings)} keys are in your own MODEL but NOT in the MAPPINGS:")
        for k in sorted(list(model_vs_mappings)):
            print(f"  + {k}")

    with torch.inference_mode():
        with safe_open(str(model_file), framework="pt", device="cpu") as f:
            f_keys = set(f.keys())

            file_vs_mappings = f_keys - mappings_keys
            mappings_vs_file = mappings_keys - f_keys

            if file_vs_mappings:
                print(f"\n[ACTION NEEDED] The following {len(file_vs_mappings)} keys are in the FILE but NOT in your MAPPING:")
                for k in sorted(list(file_vs_mappings)):
                    print(f"  + (In FILE) {k}")

            if mappings_vs_file:
                print(f"\n[ACTION NEEDED] The following {len(mappings_vs_file)} keys are in the MAPPING but NOT in the FILE:")
                for k in sorted(list(mappings_vs_file)):
                    print(f"  + (In MAPPING) {k}")

            step = 0
            for src_key, dst_path in all_mappings.items():
                assert src_key in f_keys, f"Missing key in checkpoint: {src_key}"

                t = f.get_tensor(src_key)

                if any(src_key.endswith(suffix) for suffix in needs_T):
                    t = t.T.contiguous()

                obj = model
                parts = dst_path.split('.')
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                name = parts[-1]

                existing = getattr(obj, name)
                dst_shape = existing.shape if hasattr(existing, "shape") else None

                if dst_shape is not None and tuple(dst_shape) != tuple(t.shape):
                    raise ValueError(
                        f"Shape mismatch for '{src_key}' -> '{dst_path}': "
                        f"checkpoint {tuple(t.shape)} vs model {tuple(dst_shape)}"
                    )

                real = torch.empty_like(t, device=device, dtype=torch_dtype)
                real.copy_(t, non_blocking=True)

                setattr(obj, name, torch.nn.Parameter(real, requires_grad=False))

                del t
                step += 1
                if (step % 8) == 0:
                    gc.collect()

    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()

    return model
