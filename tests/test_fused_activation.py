"""HyperFlow's bypass must see the same SwiGLU input as ComfyUI's fc2."""
import pytest
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
from comfy.ldm.minimax.model import MLP
from comfy.weight_adapter.bypass import BypassForwardHook
from hyperflow_h3.apply import _FrugalLoRA


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ffn,rank,tokens", [(16, 4, 3), (14336, 256, 321)])
def test_h3_mlp_fused_swiglu_bypass(monkeypatch, device, dtype, ffn, rank, tokens):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setattr(comfy.model_management, "get_torch_device",
                        lambda: torch.device(device))
    torch.manual_seed(42)
    mlp = MLP(8, ffn, dtype=dtype, device=device, operations=comfy.ops.manual_cast)
    mlp.fc1.weight.data.normal_(std=0.1)
    mlp.fc2.weight.data.normal_(std=0.01)
    down = torch.randn(rank, ffn, device=device, dtype=dtype) * 0.01
    up = torch.randn(8, rank, device=device, dtype=dtype) * 0.01
    adapter = _FrugalLoRA(set(), (up, down, float(rank), None, None, None))
    hook = BypassForwardHook(mlp.fc2, adapter, multiplier=0.7)
    x = torch.randn(tokens, 8, device=device, dtype=dtype)
    gate, value = F.linear(x, mlp.fc1.weight).chunk(2, dim=-1)
    activated = F.silu(gate) * value
    want = F.linear(activated, mlp.fc2.weight) + 0.7 * F.linear(F.linear(activated, down), up)
    original = mlp.fc2.forward
    hook.inject()
    try:
        got = mlp(x)
        # Positional activation arguments follow the same ComfyUI Linear contract.
        positional = mlp.fc2(F.linear(x, mlp.fc1.weight), "swiglu")
        # Older H3 versions apply SwiGLU before calling fc2.
        eager = mlp.fc2(activated)
    finally:
        hook.eject()
    tolerance = 0.02 if dtype == torch.bfloat16 else 1e-5
    for result in (got, positional, eager):
        torch.testing.assert_close(result, want, atol=tolerance, rtol=tolerance)
    assert mlp.fc2.forward == original
