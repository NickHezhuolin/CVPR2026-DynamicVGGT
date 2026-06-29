<div align="center">
<h1>DynamicVGGT: Learning Dynamic Point Maps for 4D Scene Reconstruction in Autonomous Driving</h1>

<p><strong>CVPR 2026</strong> &nbsp;·&nbsp; Built on <a href="https://github.com/facebookresearch/vggt">VGGT</a> (CVPR 2025)</p>

<p>
Zhuolin He, Jing Li, Guanghao Li, Xiaolei Chen, Jiacheng Tang, Siyang Zhang, Zhounan Jin, Feipeng Cai, Bin Li, Jian Pu, et al.
</p>

<a href="https://arxiv.org/abs/2603.08254">Paper</a> &nbsp;·&nbsp;
<a href="https://github.com/NickHezhuolin/DynamicVGGT">GitHub</a> &nbsp;·&nbsp;
<a href="https://github.com/facebookresearch/vggt">VGGT</a> &nbsp;·&nbsp;
<a href="training/README.md">Training</a>
</div>

```bibtex
@inproceedings{he2026dynamicvggt,
  title={Dynamicvggt: Learning dynamic point maps for 4d scene reconstruction in autonomous driving},
  author={He, Zhuolin and Li, Jing and Li, Guanghao and Chen, Xiaolei and Tang, Jiacheng and Zhang, Siyang and Jin, Zhounan and Cai, Feipeng and Li, Bin and Pu, Jian and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={35670--35679},
  year={2026}
}
```

## Updates

- **[Jun 2026]** Code release for DynamicVGGT. This repository provides model and training code only — **pretrained weights are not included**. Use your own checkpoint for inference or fine-tuning.

## Overview

**DynamicVGGT** extends [VGGT](https://github.com/facebookresearch/vggt) with **multiview + temporal** modeling for dynamic scenes. Given a sequence of multi-camera observations, the model predicts camera parameters, depth maps, 3D point maps, future point maps, and (optionally) Gaussian-splatting outputs.

Compared to the original VGGT feed-forward pipeline, DynamicVGGT adds:

- **ParallelAggregator** with frame, global, and temporal attention for dynamic scenes
- **Future point head** for predicting future 3D structure from temporal tokens
- **Gaussian splatting head** (optional, requires `gsplat`) for differentiable rendering

This is a **code-only** release. Pretrained DynamicVGGT checkpoints, proprietary driving datasets, and the internal `VGGTEval` harness are not bundled.

## Quick Start

First, clone this repository and install dependencies:

```bash
git clone https://github.com/NickHezhuolin/DynamicVGGT.git
cd DynamicVGGT
pip install -e .
pip install -r requirements.txt
```

Run a minimal forward pass with randomly initialized weights (no checkpoint required):

```bash
python examples/inference.py
```

Load a user-provided checkpoint:

```bash
python examples/inference.py --checkpoint /path/to/your_checkpoint.pt
```

Programmatic usage with a multiview-temporal batch:

```python
import torch
from dynamicvggt.models.dynamicvggt import DynamicVGGT

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = DynamicVGGT(
    enable_camera=True,
    enable_depth=True,
    enable_point=True,
    enable_gs=False,           # set True if gsplat is installed
    enable_future_point=True,
).to(device)
model.eval()

batch = {
    "input_images": torch.rand(1, 2, 2, 3, 518, 518, device=device),  # [B, T, V, C, H, W]
    "input_extrinsics": torch.eye(4, device=device).view(1, 1, 1, 4, 4).expand(1, 2, 2, 4, 4)[..., :3, :],
    "input_intrinsics": torch.tensor(
        [[[500.0, 0.0, 259.0], [0.0, 500.0, 259.0], [0.0, 0.0, 1.0]]],
        device=device,
    ).view(1, 1, 1, 3, 3).expand(1, 2, 2, 3, 3),
    "has_flow": torch.tensor([False], device=device),
    "has_moge_depth": torch.tensor([False], device=device),
    "time_gap": 1,
}

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(batch)

print(sorted(predictions.keys()))
# ['depth', 'depth_conf', 'extrinsic', 'future_world_points', 'future_world_points_conf',
#  'images', 'intrinsic', 'pose_enc', 'pose_enc_list', 'time_gap', 'world_points', 'world_points_conf']
```

## Detailed Usage

<details>
<summary>Click to expand</summary>

### Input format

DynamicVGGT expects a **multiview-temporal** batch when `input_extrinsics` is present:

| Key | Shape | Description |
|-----|-------|-------------|
| `input_images` | `[B, T, V, C, H, W]` | Multi-camera frames over time |
| `input_extrinsics` | `[B, T, V, 3, 4]` | Camera extrinsics (OpenCV `camera-from-world`) |
| `input_intrinsics` | `[B, T, V, 3, 3]` | Camera intrinsics |
| `has_flow` | `[B]` | Whether optical flow supervision is available |
| `has_moge_depth` | `[B]` | Whether MoGe depth supervision is available |
| `time_gap` | scalar | Temporal gap for future-point supervision (optional) |

For the static VGGT-style path (no `input_extrinsics`), use `batch["images"]` with shape `[B, S, C, H, W]`.

### Enable optional heads

```python
# Gaussian splatting (requires: pip install gsplat)
model = DynamicVGGT(enable_gs=True).to(device)
outputs = model(batch)
# outputs may include: gs_depth, gs_depth_conf, render_results

# Disable future-point prediction (falls back to standard Aggregator)
model = DynamicVGGT(enable_future_point=False).to(device)
```

### Load checkpoint

```python
state = torch.load("/path/to/checkpoint.pt", map_location=device)
state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
model.load_state_dict(state_dict, strict=False)
```

### Inference API

`model.inference(images, batch)` accepts `images` with shape `[B, T, V, C, H, W]` and runs the same prediction heads as `forward()`.

</details>

## Training

Training code lives under [`training/`](training/). See [`training/README.md`](training/README.md) for dataset setup and common questions.

### VGGT fine-tuning (Co3D baseline)

Fine-tune the upstream VGGT model on Co3D using `training/config/default.yaml`:

```bash
cd training
torchrun --nproc_per_node=4 launch.py --config default
```

Configure dataset paths and optional `checkpoint.load_pretrained_path` in the YAML file.

### DynamicVGGT training (multiview + temporal)

Train DynamicVGGT with `training/config/dynamicvggt.yaml`:

```bash
cd training
torchrun --nproc_per_node=4 launch.py --config dynamicvggt
```

You must implement dataset classes compatible with [`MultiComposedDataset`](training/data/multi_composed_dataset.py). Each sample should provide `input` / `target` tensors with shape `(T, V, H, W, ...)`, plus ego-motion metadata. Add your dataset configs under `data.train_own_dataset.dataset.dataset_configs` in `dynamicvggt.yaml`.

Optional initialization weights:

```yaml
checkpoint:
  load_pretrained_path: /path/to/your_init_weights.pt
```

## Optional Dependencies

```bash
# Gaussian-splatting heads
pip install gsplat
```

## VGGT Demos (upstream)

The original VGGT Gradio / Viser / COLMAP demos are included in this fork:

```bash
pip install -r requirements_demo.txt
python demo_gradio.py
python demo_viser.py --image_folder path/to/your/images/folder
python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/
```

See the upstream [VGGT README](https://github.com/facebookresearch/vggt) for details.

## Project Structure

```
DynamicVGGT/
├── dynamicvggt/          # DynamicVGGT model, heads, and layers
├── vggt/                 # Upstream VGGT backbone and utilities
├── training/             # Distributed trainer, dataloaders, Hydra configs
├── examples/             # Inference examples
├── demo_*.py             # VGGT interactive demos
└── requirements.txt
```

## Acknowledgements

This project builds on [VGGT](https://github.com/facebookresearch/vggt) (CVPR 2025), [DINOv2](https://github.com/facebookresearch/dinov2), [gsplat](https://github.com/nerfstudio-project/gsplat), and related open-source geometry learning work.

```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## Checklist

- [x] Release DynamicVGGT model code
- [x] Release training code and configs
- ❌ Provide inference example
- ❌ Release pretrained DynamicVGGT checkpoints

## License

See the [LICENSE](./LICENSE.txt) file for details about the license under which this code is made available.

This repository inherits the Meta VGGT Research License from the upstream fork. **Pretrained weights are not bundled.** If you use upstream VGGT checkpoints (e.g. `facebook/VGGT-1B` on Hugging Face), their usage remains subject to Meta's checkpoint terms.
