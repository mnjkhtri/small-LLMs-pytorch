import torch
import gc
from safetensors import safe_open
from utils.dump import LOGGER


def _summarize_layer_keys(keys):
    groups, other = {}, []
    for k in sorted(keys):
        parts = k.split(".")
        if len(parts) >= 3 and parts[0] == "h" and parts[1].isdigit():
            idx = int(parts[1]); rest = ".".join(parts[2:])
            groups.setdefault(rest, []).append(idx)
        else:
            other.append(k)

    def fmt_ranges(nums):
        nums = sorted(set(nums))
        if not nums: return ""
        out=[]; s=e=nums[0]
        for x in nums[1:]:
            if x == e + 1: e = x
            else: out.append(f"{s}" if s==e else f"{s}..{e}"); s=e=x
        out.append(f"{s}" if s==e else f"{s}..{e}")
        return ",".join(out)

    summarized=[]
    for rest, idxs in sorted(groups.items()):
        summarized.append(f"h.{{{fmt_ranges(idxs)}}}.{rest}")
    summarized.extend(other)
    return summarized

def _log_missing_keys(set_a, set_b, context_a, context_b, *, level="warn", max_show=8, verbose=False):
    diff = set_a - set_b
    if not diff: return
    summarized = _summarize_layer_keys(diff)
    shown = summarized if verbose else summarized[:max_show]
    more = "" if verbose or len(summarized) <= max_show else f" …(+{len(summarized)-max_show})"
    tag = {"warn":"⚠️", "info":"ℹ️", "error":"❌"}.get(level, "⚠️")
    header = f"{tag} {len(diff)} keys in {context_a} not in {context_b}:"
    if shown:
        print(header + "\n  - " + "\n  - ".join(shown) + more)
    else:
        print(header)


def stream_weights(model, checkpoint_path, param_mapping, transpose_suffixes=(), target_dtype=None, device="cpu", log_every=512, gc_every=16):
    LOGGER.block("weights: stream to model")
    LOGGER.step(f"🧩 ckpt={checkpoint_path} device={device} dtype={target_dtype if target_dtype is not None else 'model'} map={len(param_mapping)}")
    LOGGER.cuda_mem("pre ")

    mapping_values=set(param_mapping.values())
    model_keys=set(model.state_dict().keys())
    _log_missing_keys(mapping_values, model_keys, "MAPPING dst", "MODEL keys")
    _log_missing_keys(model_keys, mapping_values, "MODEL keys", "MAPPING dst")

    def resolve_dst(dst_path):
        sub = model
        parts = dst_path.split(".")
        for part in parts[:-1]:
            sub = getattr(sub, part)
        name = parts[-1]
        obj = getattr(sub, name)
        ten = obj.data if isinstance(obj, torch.nn.Parameter) else obj
        if not torch.is_tensor(ten):
            raise TypeError(f"{dst_path} is not Tensor (got {type(obj)})")
        return sub, name, obj, ten

    with LOGGER.timer() as t:
        with torch.inference_mode():
            with safe_open(str(checkpoint_path), framework="pt", device="cpu") as ckpt:
                ckpt_keys=set(ckpt.keys())
                _log_missing_keys(ckpt_keys, set(param_mapping.keys()), "CHECKPOINT", "MAPPING src")
                _log_missing_keys(set(param_mapping.keys()), ckpt_keys, "MAPPING src", "CHECKPOINT")

                total=len(param_mapping)
                LOGGER.step(f"🚚 streaming={total}")
                for i, (src_key, dst_path) in enumerate(param_mapping.items(), start=1):

                    # 1) Fetch tensor from checkpoint (CPU)
                    if src_key not in ckpt_keys:
                        raise KeyError(f"missing in checkpoint: {src_key}")
                    src = ckpt.get_tensor(src_key)

                    # 2) Optional transpose fixup for mismatched layouts
                    if transpose_suffixes and any(src_key.endswith(s) for s in transpose_suffixes):
                        src = src.T.contiguous()

                    # 3) Resolve destination path "a.b.c.weight" into:
                    sub, name, dst_obj, dst = resolve_dst(dst_path)

                    # 4) Give the Quant layers load_param_ to intercept the pack.
                    handled = False
                    lp = getattr(sub, "load_wts_", None)
                    if callable(lp):
                        handled = bool(lp(name, src))

                    # 5) If not handled by quant layer, do normal tensor copy.
                    if not handled:
                        if tuple(dst.shape) != tuple(src.shape):
                            raise ValueError(
                                f"shape mismatch {src_key}->{dst_path}: {tuple(src.shape)} vs {tuple(dst.shape)}"
                            )
                        if src.dtype != dst.dtype: src = src.to(dtype=dst.dtype)
                        if src.device != dst.device: src = src.to(device=dst.device, non_blocking=True)
                        dst.copy_(src, non_blocking = True)

                        # Freeze loaded parameters by default (common for inference)
                        if isinstance(dst_obj, torch.nn.Parameter):
                            dst_obj.requires_grad_(False)

                    # Progress tracker:
                    if log_every and (i % log_every) == 0:
                        LOGGER.step(f"📥 loaded={i}/{total}")
                        LOGGER.cuda_mem("mid ")

                    if gc_every and (i % gc_every) == 0: gc.collect()

    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    LOGGER.cuda_mem("post ")
    LOGGER.done("weights loaded", elapsed=t.dt)
    return model
