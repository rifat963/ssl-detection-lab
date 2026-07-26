# YOLO12 SSL tutorial notebooks

One Kaggle-ready notebook per self-supervised objective, each pretraining a **YOLO12** backbone
on the football player detection dataset and transferring the result into a YOLO12 detector.

| Notebook | Objective | Views | Key hyperparameters |
|---|---|---|---|
| [simclr](simclr_yolo12_football_ssl_tutorial.ipynb) | NT-Xent contrastive | 2 | `temperature`, `projection_dim` |
| [byol](byol_yolo12_football_ssl_tutorial.ipynb) | Online-to-EMA latent prediction | 2 | `momentum`, `final_momentum` |
| [moco](moco_yolo12_football_ssl_tutorial.ipynb) | InfoNCE with a negative queue | 2 | `queue_size`, `temperature` |
| [dinov2](dinov2_yolo12_football_ssl_tutorial.ipynb) | Multi-crop self-distillation | 2 + `local_crops` | `teacher_temperature`, `koleo_weight` |
| [dinov3](dinov3_yolo12_football_ssl_tutorial.ipynb) | Frozen DINOv3 teacher distillation | 1 | `dinov3_weights`, `dinov3_model` |
| [mae](mae_yolo12_football_ssl_tutorial.ipynb) | Masked pixel reconstruction | 1 | `mask_ratio` |
| [ijepa](ijepa_yolo12_football_ssl_tutorial.ipynb) | Masked latent-block prediction | 1 | `num_target_blocks`, `predictor_depth` |

## Naming: use `yolo12`, not `yolov12`

Ultralytics dropped the `v` from YOLO11 onward. The architecture file is `yolo12n.yaml`:

- `YOLO("yolov12n.yaml")` raises `FileNotFoundError` — the file does not exist upstream.
- A weights file named `yolov12n.pt` loads fine, but `resolve_model_family` does not recognize
  the spelling, so reports label it *Custom Ultralytics YOLO* instead of *YOLO12*.

Every notebook uses `yolo12n.yaml`. Scale with `yolo12n` / `yolo12s` / `yolo12m` / `yolo12l` /
`yolo12x`.

## Requirements

- Kaggle **GPU T4 x2** with Internet enabled
- The `iasadpanwhar/football-player-detection-yolov8` dataset attached via **Add Input**
- `dinov3` additionally needs PyTorch 2.7.1+ and user-supplied official Meta DINOv3 weights,
  which carry their own separate DINOv3 License terms

## What each notebook does

1. Configure the Kaggle runtime and locate the dataset
2. Build and validate a `PretrainConfig` for the objective
3. Preview the augmented views the objective actually consumes
4. Train across both GPUs with `launch_distributed_pretrain`
5. Plot the loss history and read the run manifest
6. Transfer the online/student encoder into a YOLO12 detector checkpoint

Pretraining reads image pixels only; annotation files are never opened. A falling SSL loss is not
a detection metric — fine-tune and evaluate against a scratch-trained baseline before drawing any
conclusion.
