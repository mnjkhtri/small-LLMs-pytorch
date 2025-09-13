import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from models.gpt2 import GPT2

def _has_cuda():
    return torch.cuda.is_available()

def _bf16_supported():
    return _has_cuda() and torch.cuda.is_bf16_supported()

@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
def test_parity_fp32_cuda():

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    hf  = GPT2LMHeadModel.from_pretrained("gpt2").eval().to("cuda", dtype=torch.float32)
    my  = GPT2.from_pretrained("gpt2", torch_dtype=torch.float32, device='cuda').eval()

    ids = torch.tensor([tok.encode("The quick brown fox jumps over the lazy dog.")], device="cuda")
    with torch.no_grad():
        a = hf(ids).logits[:, -1, :].float()
        b = my(ids)['logits'][:, -1, :].float()

    diff = (a - b).abs().max().item()
    assert diff < 1e-4, f"fp32 CUDA parity failed, max diff {diff}"

@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
def test_parity_fp16_cuda():

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    hf  = GPT2LMHeadModel.from_pretrained("gpt2").eval().to("cuda", dtype=torch.float16)
    my  = GPT2.from_pretrained("gpt2", torch_dtype=torch.float16, device='cuda').eval()

    ids = torch.tensor([tok.encode("The quick brown fox jumps over the lazy dog.")], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        a = hf(ids).logits[:, -1, :].float()
        b = my(ids)['logits'][:, -1, :].float()

    diff = (a - b).abs().max().item()
    assert diff < 5e-1, f"fp16 CUDA parity failed, max diff {diff}"

@pytest.mark.skipif(not _bf16_supported(), reason="CUDA bfloat16 not supported")
def test_bf16_smokes_cuda():

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    my  = GPT2.from_pretrained("gpt2", torch_dtype=torch.bfloat16, device='cuda').eval()

    ids = torch.tensor([tok.encode("bf16 smokes test!")], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = my(ids)['logits']

    assert torch.isfinite(out.float()).all(), "nan or inf in bf16 logits"
