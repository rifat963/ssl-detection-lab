# Troubleshooting

Start with `ssldet doctor`. It reports package versions against their floors, CUDA, cuDNN, and
every visible GPU — and it never crashes, so it works even when imports are broken.

## Installation and environment

**`Unsupported runtime: torch 2.1 (requires 2.4+)`**
Upgrade the package rather than PyTorch on managed platforms:
`pip install -q --upgrade "git+https://github.com/rifat963/ssl-detection-lab.git@main"`

**Training silently runs on CPU**
A generic PyTorch wheel replaced the platform's CUDA build. Check `ssldet doctor` →
`cuda.available`. Reinstall the platform's build; do not `pip install torch` on Kaggle or Colab.

**`The official DINOv3 repository requires PyTorch 2.7.1+`**
DINOv3 backbones have a higher floor than the rest of the package. `pip install "torch>=2.7.1"`
or choose a different method.

## Configuration errors

All of these come from `.validate()` and are intentional — they fire before any GPU time is spent.

| Message | Fix |
|---|---|
| `method must be one of [...]` | Check spelling; see [SSL Methods](SSL-Methods.md) |
| `image_roots must contain at least one image directory` | Non-empty list of real directories |
| `warmup_epochs must be between 0 and epochs` | Lower `warmup_epochs` |
| `Require 0 < momentum <= final_momentum <= 1` | `final_momentum` must not be below `momentum` |
| `Require 0 < teacher_temperature < student_temperature` | DINOv2: teacher must be *colder* |
| `local_crop_size must be between 32 and image_size` | Raise `image_size` or lower `local_crop_size` |
| `dinov3_weights is required` | DINOv3 needs user-supplied official Meta weights |
| `projection_dim must be divisible by predictor_heads` | I-JEPA: e.g. 128 with 4 heads |
| `I-JEPA projection_dim must also be divisible by 4` | Required by the 2-D sinusoidal embedding |
| `mask_ratio must be between 0 and 1` | MAE: exclusive bounds |

## Data

**`No supported images found under: [...]`**
Recognized suffixes: `.jpg .jpeg .png .bmp .tif .tiff .webp`. The search is recursive.

**`Image root is not a directory or does not exist`**
Usually a Kaggle dataset that was not attached via **Add Input**, or a typo'd mount path.

**`simclr requires at least two images` / `at least two images per distributed process`**
Contrastive and distillation objectives need pairs. With DDP, each rank needs two — so a tiny
dataset split across two GPUs can fail even when the total looks sufficient.

## Training

**`Non-finite SSL loss`**
Raised deliberately rather than letting NaNs poison the weights. In order: lower
`learning_rate`, lower `gradient_clip`, disable `amp`. For DINOv2 also check the temperature gap.

**CUDA out of memory**
Lower `batch_size` first (it is per GPU). Then lower `image_size`. Use `grad_accum_steps` to keep
the optimizer batch — but remember it does **not** restore contrastive negatives. For DINOv2,
lower `local_crops`.

**Loss decreases but detection does not improve**
Expected and important. SSL loss is not a detection metric. Fine-tune and evaluate against a
from-scratch baseline — see [Evaluation](Evaluation.md).

**Training hangs on multiple GPUs**
If one rank raises, the others can block at a collective until the NCCL timeout. Read the *first*
traceback in the log, not the timeout. Verify with `num_processes=1` first.

## Transfer

**`SSL-to-YOLO backbone coverage is 42.0%, below the required 95.0%`**
The target model does not match pretraining. Same family *and* same scale. See
[Downstream Transfer](Downstream-Transfer.md#coverage).

**`SSL encoder tensor shapes do not match`**
Same key names, different shapes — almost always a scale mismatch (`n` vs `s`).

**`No transferable encoder was found`**
The checkpoint has no `online_encoder.*` or `student_encoder.*` keys. Pass `encoder_prefix=` for a
custom objective, or confirm you passed an SSL checkpoint rather than a detector `.pt`.

## Models

**`FileNotFoundError: 'yolov12n.yaml' does not exist`**
Ultralytics dropped the `v` from YOLO11 onward. Use `yolo12n.yaml`. See
[Model Families](Model-Families.md#naming-no-v-from-yolo11-onward).

**Reports say "Custom Ultralytics YOLO" instead of my family**
The model name did not match a known alias — commonly a `yolov12*.pt`-style name. Functionality is
unaffected; only the reported label is wrong.

**RT-DETR / YOLO-NAS rejected for SSL**
They expose no single spatial backbone tensor. Both work for evaluation and video.

## Video

**`No video frames were produced`**
Bad path, unreachable stream, or a codec OpenCV cannot decode. Confirm the source opens in a
player, and check `source_frames` in the report.

**Per-class table shows numeric IDs instead of names**
The weights carry no class names. Fine-tuned checkpoints normally do; raw YAML-initialized ones
may not.

## Still stuck

Include `ssldet doctor` output, your `PretrainConfig`, and the full first traceback when
reporting an issue.
