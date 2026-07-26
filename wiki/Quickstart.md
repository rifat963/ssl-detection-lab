# Quickstart

Goal: pretrain a YOLO backbone without labels, then move it into a detector.

## 0. Check the environment

```bash
ssldet doctor
```

## 1. Point at unlabelled images

Any directory tree of images works. Only pixels are read — annotation files are never opened.
Supported suffixes: `.jpg .jpeg .png .bmp .tif .tiff .webp`.

```text
/data/unlabelled/
  images/
    clip_0001.jpg
    clip_0002.jpg
    ...
```

## 2. Dry-run the pipeline first

Before committing GPU hours, verify data loading, augmentation, checkpointing, and transfer with
a deliberately tiny run:

```python
from ssldet import make_dry_run_config, pretrain

config = make_dry_run_config(
    "simclr",
    train_images="/data/unlabelled/images",
    output_dir="runs/dryrun",
    max_images=32,
    image_size=128,
)
print(pretrain(config).yolo_checkpoint)
```

One epoch over 32 images proves the plumbing works. It proves nothing about model quality.

## 3. Pretrain for real

```python
from ssldet import PretrainConfig, pretrain

config = PretrainConfig(
    method="simclr",
    image_roots=["/data/unlabelled/images"],
    output_dir="runs/simclr_yolo26",
    yolo_model="yolo26n.yaml",   # .yaml = random init; .pt = COCO warm start
    epochs=25,
    batch_size=32,               # per GPU
    image_size=224,
    workers=2,
).validate()

result = pretrain(config)
```

Always call `.validate()`. It catches contradictory settings up front rather than three hours in.

### Outputs

| File | Contents |
|---|---|
| `<method>_pretrained_<model>.pt` | Detector-compatible YOLO checkpoint |
| `best_ssl.pt` / `last_ssl.pt` | Full SSL state: method, optimizer, scheduler, scaler, history |
| `history.csv` | Per-epoch loss, learning rate, EMA momentum, wall time |
| `run_manifest.json` | Provenance: method, initialization, image count, effective batch, best loss |

## 4. Two GPUs

```python
from ssldet import launch_distributed_pretrain

launch_distributed_pretrain(config, num_processes=2)
```

This shells out to `torch.distributed.run` with one process per GPU. `batch_size` is **per GPU**,
so the effective batch is `num_processes * batch_size * grad_accum_steps`.

Gradient accumulation raises the optimizer batch but does **not** create extra contrastive
negatives for SimCLR or MoCo. See [Configuration](Configuration.md#batch-size-and-accumulation).

## 5. Transfer into a detector

```python
from ssldet import transfer_ssl_backbone_to_yolo

transfer = transfer_ssl_backbone_to_yolo(
    "runs/simclr_yolo26/best_ssl.pt",
    "runs/simclr_yolo26/backbone.pt",
    yolo_model="yolo26n.yaml",
)
print(f"{transfer.coverage:.1%} coverage")   # expect 100%
```

Then fine-tune with labels as usual:

```python
from ultralytics import YOLO

YOLO("runs/simclr_yolo26/backbone.pt").train(data="dataset.yaml", epochs=50, imgsz=640)
```

## 6. Evaluate honestly

Train an identical detector from scratch and compare. Without that baseline the number means
nothing. See [Evaluation](Evaluation.md) and
[the suggested protocol](../README.md#suggested-experimental-protocol).

## Next

[SSL Methods](SSL-Methods.md) · [Configuration](Configuration.md) · [Downstream Transfer](Downstream-Transfer.md)
