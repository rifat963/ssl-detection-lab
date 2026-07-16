# SSL Detection Lab

A small, student-friendly PyTorch package for pretraining Ultralytics YOLO backbones with
**SimCLR, BYOL, MoCo, MAE, or I-JEPA**, then transferring the learned backbone directly into an
object detector.

The package was designed for CSE445 Computer Vision labs and Kaggle runtimes. Large training
loops live in normal Python modules; notebooks contain configuration, method calls, plots,
evaluation, and interpretation.

## Why the I-JEPA implementation is compute-scaled

The official I-JEPA release provides ViT-H/14 and ViT-g checkpoints. The published ViT-H/14
recipe used 16 A100 80 GB GPUs and those transformer weights do not map directly into a YOLO26
CNN backbone. This package therefore implements the defining I-JEPA learning mechanism on the
actual YOLO backbone:

1. Sample several large rectangular target blocks.
2. Remove those regions from a single context image.
3. Encode the visible context with the trainable YOLO backbone.
4. Encode the complete image with an exponential-moving-average target backbone.
5. Predict the target-block representations in latent space.
6. Transfer the trained online backbone directly into YOLO26 detection.

This is a transparent, educational adaptation. It should be described in reports as
**YOLO-native compute-scaled I-JEPA**, not as a reproduction of the official ViT-H experiment.

Official references:

- [I-JEPA paper](https://arxiv.org/abs/2301.08243)
- [Archived official I-JEPA code and checkpoints](https://github.com/facebookresearch/ijepa)
- [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/)

## Install from GitHub

After pushing this directory to a GitHub repository:

```bash
pip install "git+https://github.com/rifat963/ssl-detection-lab.git@main"
```

For active development, clone it and use an editable install:

```bash
git clone https://github.com/rifat963/ssl-detection-lab.git
cd ssl-detection-lab
pip install -e .
```

## Five methods, one interface

| Method | Learning signal | Two augmented views? | Moving target? | YOLO adaptation |
|---|---|---:|---:|---|
| SimCLR | NT-Xent contrastive loss | Yes | No | Global pooled backbone features |
| BYOL | Cross-view latent prediction | Yes | Yes | Online/EMA YOLO encoders |
| MoCo | Positive key + negative queue | Yes | Yes | Momentum YOLO key encoder |
| MAE | Masked pixel reconstruction | No | No | Lightweight CNN decoder on YOLO features |
| I-JEPA | Masked latent-block prediction | No | Yes | Transformer predictor over YOLO feature grid |

The MAE implementation is also a documented CNN-compatible adaptation rather than an exact copy
of the original ViT MAE architecture.

## Recommended dual-T4 Kaggle workflow

Create a YAML file based on `examples/ijepa_t4x2.yaml`, then launch one process per GPU:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m ssldet.cli --config /kaggle/working/ijepa_t4x2.yaml
```

Important configuration choices:

- `batch_size` is per GPU.
- `image_size: 224`, `batch_size: 16`, and `grad_accum_steps: 2` are safe starting points for
  two 16 GB T4 GPUs with YOLO26n.
- `amp: true` enables FP16 mixed precision.
- `yolo_model: yolo26n.yaml` is strict label-free SSL from random weights.
- `yolo_model: yolo26n.pt` is a practical COCO-supervised warm start followed by label-free
  football-domain adaptation. Never report this second setting as training without labels.
- Pretrain only on `train/images`; including validation or test images creates transductive
  leakage in a standard evaluation.

The output directory contains:

```text
best_ssl.pt                 # complete SSL state for analysis/resume
last_ssl.pt                 # latest complete SSL state
history.csv                 # epoch loss, learning rate, time, EMA momentum
run_manifest.json           # reproducibility and initialization record
ijepa_pretrained_yolo26.pt  # detector with the learned online backbone
```

## Single-GPU Python API

```python
from ssldet import PretrainConfig, pretrain

config = PretrainConfig(
    method="ijepa",
    image_roots=["/kaggle/input/my-dataset/train/images"],
    output_dir="/kaggle/working/ijepa_yolo26/ssl",
    yolo_model="yolo26n.yaml",
    epochs=25,
    batch_size=16,
    image_size=224,
    grad_accum_steps=2,
)

result = pretrain(config)
print(result.yolo_checkpoint)
```

Use the saved checkpoint with normal Ultralytics code:

```python
from ultralytics import YOLO

detector = YOLO("/kaggle/working/ijepa_yolo26/ssl/ijepa_pretrained_yolo26.pt")
detector.train(
    data="/kaggle/working/football_detection.yaml",
    epochs=60,
    imgsz=960,
    batch=12,
    device=[0, 1],
)
```

## Switching methods

Only the method-specific fields need to change:

```yaml
# SimCLR
method: simclr
temperature: 0.20
projection_dim: 128

# BYOL
method: byol
momentum: 0.996

# MoCo
method: moco
temperature: 0.20
queue_size: 16384

# MAE
method: mae
mask_ratio: 0.60

# I-JEPA
method: ijepa
num_target_blocks: 4
target_scale_min: 0.10
target_scale_max: 0.25
predictor_depth: 2
```

## Suggested experimental protocol

For a defensible label-efficiency study, keep the downstream detector settings identical and
compare:

1. Random YOLO26 initialization.
2. Official COCO-pretrained YOLO26.
3. Strict SSL from `yolo26n.yaml`.
4. COCO warm start plus SSL domain adaptation.

Repeat each condition with at least three seeds and report mean ± standard deviation for
mAP50-95, per-class recall, training time, peak GPU memory, and inference latency. A two-epoch
FAST run is a pipeline check, not research evidence.

## Development checks

```bash
pip install -e ".[dev]"
pytest -q
python -m compileall -q src tests
```

## License

MIT. Ultralytics and any downloaded pretrained checkpoints retain their own licenses.
