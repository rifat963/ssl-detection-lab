# Configuration

Everything about a pretraining run lives in one `PretrainConfig`. Always call `.validate()` — it
fails fast on contradictory settings instead of three hours into a job.

```python
from ssldet import PretrainConfig

config = PretrainConfig(method="simclr", image_roots=["/data/images"]).validate()
```

Load and save YAML:

```python
config = PretrainConfig.from_yaml("config.yaml")
config.save_yaml("runs/exp/config.yaml")
```

## Core fields

| Field | Default | Meaning |
|---|---|---|
| `method` | `"ijepa"` | One of the seven objectives |
| `image_roots` | `[]` | Directories searched recursively for images |
| `output_dir` | `/kaggle/working/ssldet_pretraining` | Checkpoints, history, manifest |
| `yolo_model` | `"yolo26n.yaml"` | `.yaml` = random init, `.pt` = COCO warm start |
| `epochs` | 25 | |
| `batch_size` | 16 | **Per GPU** |
| `image_size` | 224 | Must be ≥ 32 |
| `workers` | 2 | DataLoader workers |
| `max_images` | `None` | Deterministically subsample, seeded by `seed` |
| `seed` | 42 | Offset per rank so ranks differ |

Head sizing, shared by every objective with a projection head:

| Field | Default | Meaning |
|---|---|---|
| `projection_dim` | 256 | Output width of the projection head (bottleneck for DINOv2) |
| `hidden_dim` | 1024 | Hidden width inside the projection MLP |

## Optimization

| Field | Default | Rule |
|---|---|---|
| `learning_rate` | 3e-4 | > 0 |
| `min_learning_rate` | 3e-6 | `0 ≤ min ≤ learning_rate` — cosine floor |
| `weight_decay` | 1e-4 | ≥ 0 |
| `warmup_epochs` | 1 | `0 ≤ warmup ≤ epochs` |
| `grad_accum_steps` | 1 | ≥ 1 |
| `gradient_clip` | 5.0 | > 0 |
| `amp` | `True` | Only active on CUDA |

The schedule is linear warmup then cosine decay to `min_learning_rate`. Non-finite losses raise
`FloatingPointError` immediately rather than silently poisoning the weights.

## Batch size and accumulation

`batch_size` is **per GPU**. The effective optimizer batch is:

```text
world_size * batch_size * grad_accum_steps
```

Recorded as `effective_batch_size` in `run_manifest.json`.

> **Important.** Gradient accumulation does **not** create extra contrastive negatives for SimCLR
> or MoCo. It increases the optimizer batch only. If you need more negatives, raise `batch_size`
> or (for MoCo) `queue_size`.

Accumulation groups that do not divide evenly are handled correctly — the final short group is
divided by its true size, not by `grad_accum_steps`.

## Method-specific fields

Validation only enforces a field's rules when it applies to the selected `method`.

### SimCLR / MoCo
| Field | Default | Rule |
|---|---|---|
| `temperature` | 0.2 | > 0 |
| `queue_size` | 16384 | MoCo only, ≥ 1 |

### BYOL / MoCo / DINOv2 / I-JEPA (EMA targets)
| Field | Default | Rule |
|---|---|---|
| `momentum` | 0.996 | `0 < momentum ≤ final_momentum ≤ 1` |
| `final_momentum` | 1.0 | Cosine-annealed toward across training |

### DINOv2
| Field | Default | Rule |
|---|---|---|
| `dino_output_dim` | 4096 | ≥ 2 |
| `student_temperature` | 0.10 | |
| `teacher_temperature` | 0.04 | `0 < teacher < student` |
| `center_momentum` | 0.90 | `0 ≤ x < 1` |
| `koleo_weight` | 0.10 | ≥ 0 |
| `local_crops` | 4 | ≥ 0 |
| `local_crop_size` | 96 | `32 ≤ x ≤ image_size` |

### DINOv3
| Field | Default | Rule |
|---|---|---|
| `dinov3_model` | `dinov3_vits16` | Must be a known spec |
| `dinov3_weights` | `None` | **Required** |
| `dinov3_repository` | `facebookresearch/dinov3` | Hub repo, or a local clone path when `dinov3_source="local"` |
| `dinov3_source` | `github` | `github` or `local` |
| `dinov3_global_weight` | 1.0 | ≥ 0 |
| `dinov3_dense_weight` | 1.0 | ≥ 0, and the two must not both be 0 |

`image_size` must be divisible by the teacher patch size (16 for ViT/16, 32 for ConvNeXt).

### MAE
| Field | Default | Rule |
|---|---|---|
| `mask_ratio` | 0.60 | `0 < x < 1` |

### I-JEPA
| Field | Default | Rule |
|---|---|---|
| `num_target_blocks` | 4 | ≥ 1 |
| `target_scale_min/max` | 0.10 / 0.25 | `0 < min ≤ max < 1` |
| `target_aspect_min/max` | 0.75 / 1.50 | `0 < min ≤ max` |
| `predictor_depth` | 2 | ≥ 1 |
| `predictor_heads` | 4 | ≥ 1 |
| `projection_dim` | 256 | Divisible by 4 **and** by `predictor_heads` |

## Checkpointing and resuming

| Field | Default | Meaning |
|---|---|---|
| `save_every` | 1 | Epoch interval for `last_ssl.pt` |
| `resume` | `None` | Path to an SSL checkpoint |

Resuming restores method weights, optimizer, scheduler, gradient scaler, epoch, global step, best
loss, and history. Checkpoints load with `weights_only=True`.

```python
PretrainConfig(..., resume="runs/exp/last_ssl.pt").validate()
```

`best_ssl.pt` is written whenever the reduced epoch loss improves. Under DDP the loss is averaged
across ranks before comparison, so every rank agrees on what "best" means.

## Next

[SSL Methods](SSL-Methods.md) · [Quickstart](Quickstart.md) · [Troubleshooting](Troubleshooting.md)
