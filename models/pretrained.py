from dataclasses import dataclass
import torch
import torch.nn as nn
from utils.dump import LOGGER
from utils.io import download_safetensors
from utils.stream import stream_weights
from utils.quant import QUANT_MAP


@dataclass(frozen=True)
class PretrainedSpec:
    url: str
    cache_dir: str
    config: dict
    mappings: dict
    needs_T: tuple = ()


class FromPretrainedMixin:
    PRETRAINED_SPECS = {}

    @classmethod
    def from_pretrained(cls, model_type="BASE", *, torch_dtype, device, quantize=None):
        LOGGER.block("from_pretrained")
        LOGGER.step(f"🏷️ class={cls.__name__} type={model_type} dtype={torch_dtype} device={device} q={quantize or 'none'}")
        LOGGER.cuda_mem("entry ")

        LOGGER.block("spec")
        try:
            spec = cls.PRETRAINED_SPECS[model_type]()
        except KeyError:
            raise ValueError(f"{cls.__name__} has no spec for model_type={model_type}. Available: {list(cls.PRETRAINED_SPECS)}")

        LOGGER.step(f"🔗 url={spec.url} cache_dir={spec.cache_dir}")
        LOGGER.step(f"🗝️ config_keys={list(spec.config.keys())}")
        LOGGER.step(f"🧭 mapping={len(spec.mappings)} needs_T={spec.needs_T or 'none'}")
        LOGGER.done("spec ok")

        model_file = download_safetensors(spec.url, spec.cache_dir)
        load_device = device

        LOGGER.block("meta init")
        with LOGGER.timer() as t_meta:
            with torch.device("meta"):
                model = cls(**spec.config)
                model.to(torch_dtype)

                qcls = None if (quantize is None) else QUANT_MAP.get(quantize, None)
                if quantize is not None and qcls is None:
                    raise ValueError(f"Unknown quantize='{quantize}'. Available: {sorted(QUANT_MAP.keys())}")
        
                if qcls is not None:
                    replaced = 0
                    def rec(parent: nn.Module, prefix=""):
                        nonlocal replaced
                        for name, child in list(parent.named_children()):
                            path=f"{prefix}.{name}" if prefix else name
                            if qcls.is_ok(child, path):
                                setattr(parent, name, qcls.build(child, torch_dtype))
                                replaced += 1
                            else:
                                rec(child, path)
                    rec(model)
                    LOGGER.step(f"🧊 replaced_total={replaced}")

        LOGGER.done("meta ok", elapsed=t_meta.dt)

        LOGGER.block("materialize")
        LOGGER.cuda_mem("pre ")
        with LOGGER.timer() as t_mat:
            model.to_empty(device=load_device)

        LOGGER.cuda_mem("post ")
        LOGGER.done("materialized", elapsed=t_mat.dt)

        # log_block("model")
        # print(model)
        # log_done("printed")

        model = stream_weights(model, model_file, spec.mappings, list(spec.needs_T), torch_dtype, load_device)

        LOGGER.step("finalize")
        try: LOGGER.step(f"🧮 params={sum(p.numel() for p in model.parameters()):,}")
        except Exception: pass
        LOGGER.cuda_mem("final ")
        LOGGER.done("from_pretrained ok", gap=2)
        
        LOGGER.block("generation")
        return model
