"""Retained scene patchers must not keep inactive bypass tensors on CUDA."""
from types import SimpleNamespace

import pytest
import torch

import comfy.model_management
from comfy.weight_adapter.bypass import BypassForwardHook
from hyperflow_h3.apply import _FrugalLoRA, _SplitQKVLoRA, _install_injection


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("split_qkv", [False, True])
def test_retained_scene_adapters_release_cuda(monkeypatch, split_qkv):
    monkeypatch.setattr(comfy.model_management, "get_torch_device",
                        lambda: torch.device("cuda"))
    owner = torch.nn.Linear(64, 192 if split_qkv else 64, bias=False, device="cuda")
    retained = []
    x = torch.randn(2, 64, device="cuda")
    original = owner.forward
    try:
        for _ in range(4):
            if split_qkv:
                adapter = _SplitQKVLoRA(set(), (
                    torch.randn(3, 64, 8), torch.randn(24, 64), 24.0,
                    None, None, None))
            else:
                adapter = _FrugalLoRA(set(), (
                    torch.randn(64, 8), torch.randn(8, 64), 8.0, None, None, None))
            hook = BypassForwardHook(owner, adapter)
            patcher = SimpleNamespace(model=owner, injections={})
            patcher.set_injections = lambda k, v, p=patcher: p.injections.update({k: v})
            _install_injection(patcher, [hook])
            injection = patcher.injections["hyperflow_lora"][0]
            retained.append((patcher, injection, hook))
            injection.inject(patcher)
            assert hook.adapter.weights[0].device.type == "cuda"
            # Superseded scenes stay alive in the recursive execution cache.
            for _, _, old_hook in retained[:-1]:
                assert old_hook.adapter.weights[0].device.type == "cpu"
            expected = owner(x).detach().clone()
            injection.eject(patcher)
            assert owner.forward == original
            assert hook.adapter.weights[0].device.type == "cpu"
            injection.inject(patcher)
            torch.testing.assert_close(owner(x), expected)
    finally:
        for patcher, injection, hook in reversed(retained):
            injection.eject(patcher)
