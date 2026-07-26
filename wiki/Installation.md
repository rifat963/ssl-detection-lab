# Installation

## Requirements

| Component | Minimum | Notes |
|---|---|---|
| Python | 3.10 | |
| PyTorch | 2.4 | 2.7.1+ required for DINOv3 backbones |
| torchvision | 0.19 | transforms v2 pipeline |
| Ultralytics | 8.4.96 | AGPL-3.0; see [licensing](Home.md#licensing) |

## Install

```bash
pip install "git+https://github.com/rifat963/ssl-detection-lab.git@main"
```

Editable install for development:

```bash
git clone https://github.com/rifat963/ssl-detection-lab.git
cd ssl-detection-lab
pip install -e ".[dev]"
```

## Optional extras

```bash
pip install "ssl-detection-lab[dinov3]"        # PyTorch 2.7.1+ floor for DINOv3 backbones
pip install "ssl-detection-lab[dinov3-yolo]"   # LightlyTrain DINOv3 -> YOLO26 distillation
pip install "ssl-detection-lab[dev]"           # pytest + ruff
```

## Verify the environment

Always run this before a long job:

```bash
ssldet doctor
```

It reports Python, platform, package versions against their floors, CUDA runtime, cuDNN, and
every visible GPU with compute capability and memory. It is deliberately crash-proof: a broken
CUDA driver or DLL is reported as an error field rather than raising, so you can still read the
diagnosis.

Programmatic equivalent:

```python
from ssldet import assert_supported_runtime, runtime_report

print(runtime_report())
assert_supported_runtime(require_cuda=True, minimum_gpus=2)   # raises if unmet
```

## Managed GPU platforms (Kaggle, Colab, SageMaker)

Keep the platform's CUDA-matched PyTorch build. Upgrade **this package and Ultralytics only**:

```bash
pip install -q --upgrade "git+https://github.com/rifat963/ssl-detection-lab.git@main"
```

Do not install a generic PyTorch wheel on top — that commonly replaces a CUDA build with a
CPU-only one and silently drops you to CPU training.

## DINOv3 weights

DINOv3 backbones need official weights from Meta, supplied by you as a local path or an
authorized URL. Supplying them does not change their separate
[DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) terms.

```python
from ssldet import load_dinov3_backbone

encoder = load_dinov3_backbone("dinov3_vitb16", weights="/path/to/weights.pth", device="cuda")
```

## Next

[Quickstart](Quickstart.md) · [Troubleshooting](Troubleshooting.md)
