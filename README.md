# Hyperflow Fixed for ComfyUI

A community maintenance copy of the original **ComfyUI-Hyperflow** nodes by **Drbaph / Saganaki22**, with a compatibility fix for the ComfyUI update installed on **7 October 2026**.

HyperFlow runs MiniMax-H3 video and audio generation using its 8-step adapter. This repository includes the complete node pack, example workflows, conversion tool, and tests. Original credits and the Apache-2.0 license are preserved.

## What this fixes

Hyperflow's bypass LoRA could crash during text preprocessing or sampling with:

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (321x28672 and 14336x256)
```

ComfyUI's native MiniMax-H3 MLP now calls its second linear layer with `input_act="swiglu"`. The base layer applies SwiGLU internally, reducing the last dimension from `28672` to `14336`. Hyperflow's LoRA branch was still projecting the input before that activation.

The fix in [`hyperflow_h3/apply.py`](hyperflow_h3/apply.py) applies ComfyUI's own activation helper to the LoRA input. The base model keeps its normal forward path, and calls that already apply SwiGLU before the linear layer continue to work. Node names and workflow connections are unchanged; no ComfyUI core files need editing.

## Installation

### Existing Hyperflow installation

1. Stop ComfyUI and back up your existing `custom_nodes/ComfyUI-Hyperflow` folder.
2. Download this repository using **Code → Download ZIP**.
3. Extract it and copy its contents into your existing `custom_nodes/ComfyUI-Hyperflow` folder, replacing the corresponding files. The `__init__.py` file must be directly inside that folder.
4. Remove the temporary extracted folder from `custom_nodes` if you placed it there, so ComfyUI loads only one Hyperflow installation.
5. Restart ComfyUI completely and rerun your workflow.

For the smallest update, replace only `hyperflow_h3/apply.py`; that is the only runtime file changed by this compatibility fix.

### New installation

From your ComfyUI directory:

```bash
git clone https://github.com/Nebuluss/Hyperflow_fixed.git custom_nodes/ComfyUI-Hyperflow
```

Then restart ComfyUI. Do not install a second copy alongside an existing Hyperflow folder.

## Models and usage

Use a compatible MiniMax-H3 base and the converted Hyperflow node weights from [drbaph/Hyperflow-Comfyui](https://huggingface.co/drbaph/Hyperflow-Comfyui). Put the weights in `ComfyUI/models/hyperflow/`.

- Full bases use `custom_node_hyperflow_8step_v1.0_comfyui.safetensors`.
- Pruned/curve bases use `custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors`, which applies the backbone adapter with single-time conditioning.

Wire the `SIGMAS` output of `ApplyHyperFlow` into `SamplerCustomAdvanced`; Euler is the original 8-step recipe. Existing workflows and settings can be kept when updating to this fix.

See the [original English documentation](README_UPSTREAM.md), [Chinese documentation](README_ZH.md), and [`example_workflows/`](example_workflows) for full setup details, conditioning, and optional sparse attention settings.

## Validation

The patched node pack was tested against the locally installed ComfyUI with Python 3.12 and PyTorch `2.13.0+cu130`:

- **66 tests passed; 1 skipped.** The skipped test requires Windows symlink creation.
- CPU and CUDA checks in float32 and bfloat16.
- Regression coverage for the reported dimensions: 321 tokens, width `28672 → 14336`, and LoRA rank 256.
- Both keyword and positional activation arguments, plus the older pre-activated calling path.
- Existing adapter, injection lifecycle, two-time conditioning, and model-routing tests.

The affected user has tested the patched nodes in their ComfyUI workflow and confirmed that the fix works. The automated tests additionally verify the reproduced adapter failure and tested model paths.

To run the tests with ComfyUI's Python environment, from the ComfyUI directory:

```bash
python -m pytest custom_nodes/ComfyUI-Hyperflow/tests -q
```

`pytest` is required for testing, but is not a runtime dependency of the nodes.

## Credits and license

- Original node pack: **Drbaph / Saganaki22**, [ComfyUI-Hyperflow](https://github.com/Saganaki22/ComfyUI-Hyperflow).
- HyperFlow adapter and reference implementation: [Video Rebirth / HyperFlow](https://github.com/Video-Rebirth/hyperflow).
- Base model: [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3).
- ComfyUI: [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI).
- This maintenance repository: **Nebuluss**.

Code is licensed under [Apache-2.0](LICENSE). Model weights retain their separate MiniMax-H3 model license. This repository does not include model weights or publish to the original author's ComfyUI registry account.
