# SSL Detection Lab

A small, student-friendly PyTorch package for pretraining Ultralytics YOLO backbones with
**SimCLR, BYOL, MoCo, DINOv2, MAE, or I-JEPA**, transferring the learned backbone into an object
detector, evaluating it on labelled data, and analysing local or linked videos.

The package was designed for CSE445 Computer Vision labs and Kaggle runtimes. Large training
loops live in normal Python modules; notebooks contain configuration, method calls, plots,
evaluation, and interpretation.

## Current runtime baseline

Version 0.5 uses the current accelerator-neutral PyTorch APIs and the recommended torchvision
transforms v2 pipeline. Its tested dependency floor is Python 3.10+, PyTorch 2.4+,
torchvision 0.19+, and Ultralytics 8.4.96+. On managed GPU platforms, keep the platform's
CUDA-matched PyTorch build and upgrade this package plus Ultralytics; do not blindly replace
PyTorch with a CPU-only wheel.

The baseline was refreshed on 17 July 2026 against PyTorch 2.13 documentation, torchvision
0.28 documentation, and Ultralytics 8.4.96. The lower PyTorch/torchvision floor intentionally
supports CUDA-matched Kaggle images while using the same non-deprecated APIs.

## Reusable package structure

```text
ssldet.ssl          reusable objectives, losses, interfaces, and custom-module registry
ssldet.backbones    YOLO, DINOv2, and DINOv3 feature encoders
ssldet.detection    backend-neutral detector protocol and Ultralytics adapter
ssldet.evaluation   labelled object-detection evaluation and metric export
ssldet.video        streaming detection/tracking and video metrics
ssldet.workflow     student-friendly dry-run and distributed-launch helpers
```

Use a built-in SSL objective with any compatible PyTorch encoder:

```python
from ssldet.ssl import available_ssl_modules, create_ssl_module

print(available_ssl_modules())
simclr = create_ssl_module(
    "simclr",
    encoder=my_encoder,
    feature_dim=768,
    hidden_dim=1024,
    projection_dim=256,
    temperature=0.2,
)
loss = simclr((first_view, second_view))
```

Custom objectives can subclass `SSLMethod` and be registered with `register_ssl_module()`.
Pooled objectives require an encoder returning `B x C`; MAE and I-JEPA additionally require
`forward_feature_map()` returning `B x C x H x W`.

Inspect the exact environment before a run:

```bash
ssldet doctor
```

```python
from ssldet import assert_supported_runtime, runtime_report

print(runtime_report())
assert_supported_runtime(require_cuda=True, minimum_gpus=2)
```

## Kaggle football tutorial

The student-oriented
[`ssl_detection_lab_football_t4x2_tutorial.ipynb`](output/jupyter-notebook/ssl_detection_lab_football_t4x2_tutorial.ipynb)
uses the `iasadpanwhar/football-player-detection-yolov8` dataset and a Kaggle T4 x2 runtime. It
audits the YOLO labels, dry-runs every SSL method, probes every supported model family,
fine-tunes one detector, exports labelled evaluation metrics, and exercises video analysis. The
notebook installs the library from an attached source folder when available, otherwise from the
project's GitHub repository. It contains no embedded wheel or generated Base64 package data.

## Start here: supported models and SSL architectures

Run this before starting an experiment:

```bash
ssldet models
# or, for a UI/notebook
ssldet models --json
```

### SSL architectures

| Architecture | Type | Learning objective | Views | EMA/momentum target |
|---|---|---|---:|---:|
| SimCLR | Contrastive | NT-Xent cross-view agreement | 2 | No |
| BYOL | Non-contrastive | Online-to-target latent prediction | 2 | Yes |
| MoCo | Contrastive | Positive key and negative queue | 2 | Yes |
| DINOv2 | Self-distillation | Multi-crop teacher/student agreement + KoLeo | 2+ | Yes |
| MAE | Generative | Masked pixel reconstruction | 1 | No |
| I-JEPA | Predictive | Masked latent-block prediction | 1 | Yes |

### Model families

| Family | Common scales | SSL pretraining | Labelled evaluation | Video analysis |
|---|---|---:|---:|---:|
| YOLO26 | n, s, m, l, x (+ P2/P6 YAML) | Yes | Yes | Yes |
| YOLO12 | n, s, m, l, x | Yes | Yes | Yes |
| YOLO11 | n, s, m, l, x | Yes | Yes | Yes |
| YOLOv10 | n, s, m, b, l, x | Yes | Yes | Yes |
| YOLOv9 | t, s, m, c, e | Yes | Yes | Yes |
| YOLOv8 | n, s, m, l, x | Yes | Yes | Yes |
| YOLOv6 | n, s, m, l, x | Yes | Yes | Yes |
| YOLOv5u | n, s, m, l, x | Yes | Yes | Yes |
| YOLOv3u | standard, SPP, tiny | Yes | Yes | Yes |
| Custom Ultralytics YOLO | custom | Runtime checked | Yes | Yes |
| RT-DETR | l, x | No | Yes | Yes |
| YOLO-NAS | s, m, l | No | Yes | Yes |

SSL compatibility requires an Ultralytics YAML-defined backbone ending in a spatial
`B x C x H x W` feature map. Custom models are checked when the backbone adapter is created.
YOLOv5 support means current Ultralytics-compatible **YOLOv5u** weights, not checkpoint files
from the legacy YOLOv5 repository. Evaluation and video analysis support is intentionally wider
than SSL pretraining support.

The Python equivalent is:

```python
from ssldet import capabilities

catalog = capabilities()
print(catalog["model_families"])
print(catalog["ssl_architectures"])
print(catalog["dinov2_feature_backbones"])
print(catalog["dinov3_feature_backbones"])
print(catalog["object_detection_backends"])
```

For a small one-epoch pipeline check, the public workflow helpers keep notebook code short:

```python
from ssldet import launch_distributed_pretrain, make_dry_run_config

config = make_dry_run_config(
    method="dinov2",
    train_images="/kaggle/input/.../train/images",
    output_dir="/kaggle/working/runs/dinov2_yolo11n",
    yolo_model="yolo11n.yaml",
)
result = launch_distributed_pretrain(config, num_processes=2)
print(result.succeeded, result.output_dir)
```

Use `pretrain(config)` for a normal single-process Python run. The distributed helper saves the
same validated configuration to YAML and launches one process per requested GPU.

Official DINOv2 reference feature backbones are catalogued in S/14, B/14, L/14, and g/14
sizes, each with and without register tokens. They are general-purpose ViT feature extractors,
not standalone object detectors. A detection head is required before they can produce boxes or
tracks in the video-analysis module.

**Yes, DINOv2 includes pretrained weights.** `load_dinov2_backbone()` defaults to
`pretrained=True`, so PyTorch Hub downloads Meta's official pretrained backbone. Set
`pretrained=False` for random weights, or provide `weights=`/`weights_file=` with a local
checkpoint. The standard DINOv2 code and weights are Apache-2.0 according to the
[official DINOv2 repository](https://github.com/facebookresearch/dinov2).

Load an official pretrained backbone for global embeddings or dense patch features:

```python
from ssldet import build_dinov2_transform, load_dinov2_backbone

encoder = load_dinov2_backbone("dinov2_vitb14", device="cuda")
preprocess = build_dinov2_transform(518)
# images = torch.stack([preprocess(pil_image), ...]).to("cuda")
global_embeddings = encoder(images)                    # B x 768
patch_tokens = encoder.forward_tokens(images)          # B x N x 768
feature_maps = encoder.forward_feature_map(images)     # B x 768 x H/14 x W/14
```

For offline loading, clone the official repository and provide a local checkpoint:

```python
encoder = load_dinov2_backbone(
    "dinov2_vits14_reg",
    repository="/path/to/dinov2",
    source="local",
    weights_file="/path/to/teacher_checkpoint.pth",
)
```

The loader supports:

| Model name | Architecture | Embedding | Registers |
|---|---|---:|---:|
| `dinov2_vits14` / `dinov2_vits14_reg` | ViT-S/14 | 384 | Optional |
| `dinov2_vitb14` / `dinov2_vitb14_reg` | ViT-B/14 | 768 | Optional |
| `dinov2_vitl14` / `dinov2_vitl14_reg` | ViT-L/14 | 1024 | Optional |
| `dinov2_vitg14` / `dinov2_vitg14_reg` | ViT-g/14 | 1536 | Optional |

The DINOv2 loss module itself accepts any PyTorch encoder returning a `B x C` representation.
The packaged end-to-end pretraining pipeline currently supplies adapters for the YOLO model
families in the table above. ResNet, ConvNeXt, EfficientNet, Swin, and other ViT families can
use the loss module after providing a compatible pooled-feature adapter.

### DINOv3 feature backbones and pretrained weights

DINOv3 support is a reusable **feature-backbone loader**, not a claim that this package
reproduces Meta's full DINOv3 training recipe. The official recipe includes DINO
self-distillation, iBOT, KoLeo, Gram anchoring, and large-scale FSDP2 training.

Official DINOv3 weights require access through Meta. Supply the authorized checkpoint URL or a
downloaded local file through `weights=`:

```python
from ssldet import build_dinov3_transform, load_dinov3_backbone

encoder = load_dinov3_backbone(
    "dinov3_vitb16",
    weights="/kaggle/input/dinov3-weights/dinov3_vitb16.pth",
    device="cuda",
)
preprocess = build_dinov3_transform(256, weights_dataset="lvd1689m")

global_embeddings = encoder(images)                # B x 768
patch_tokens = encoder.forward_tokens(images)      # B x N x 768
feature_maps = encoder.forward_feature_map(images) # B x 768 x H/16 x W/16
```

For offline use, clone the official repository and set `repository=/path/to/dinov3` plus
`source="local"`. Passing no `weights` constructs a randomly initialized architecture. The
official repository requires PyTorch 2.7.1+; install the optional requirement only with a
CUDA-compatible PyTorch build:

```bash
pip install -e ".[dinov3]"
```

The loader covers the official pretrained ViT-S/S+/B/L/H+/7B and ConvNeXt
Tiny/Small/Base/Large backbones. Web weights use `weights_dataset="lvd1689m"`; satellite
ViT-L/7B weights use `weights_dataset="sat493m"`. DINOv3 code and weights retain Meta's
separate [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).

## Reusable object-detection module

The object-detection adapter isolates third-party model loading from evaluation and video
reporting:

```python
from ssldet.detection import load_detector

detector = load_detector("yolo26n", "yolo26n.pt")
predictions = detector.predict(source="match.jpg", conf=0.25)
validation = detector.validate(data="football.yaml", split="test")
tracks = detector.track(source="match.mp4", tracker="botsort.yaml", stream=True)
```

`ObjectDetector` is a protocol, so another backend can implement `predict()`, `track()`, and
`validate()` without changing the SSL modules. Register it with `register_detection_backend()`
and create it through `create_detector(..., backend="name")`.

## Analyse a video link

Provide the video source, model name, and detector weight file:

```bash
ssldet video \
  --source "https://youtu.be/LNwODJXcvt4" \
  --model yolo26n \
  --weights /path/to/best.pt \
  --output runs/football-video
```

The source may be a local video, direct HTTP URL, RTSP stream, webcam index, or a video-service
URL supported by the installed Ultralytics version. Use `--tracker bytetrack.yaml` to switch
trackers, `--tracker none` for detection only, `--stride 2` to sample every second frame, or
`--max-frames 500` for a bounded run.

```python
from ssldet import VideoAnalysisConfig, analyze_video

outcome = analyze_video(
    VideoAnalysisConfig(
        video_source="https://example.com/match.mp4",
        model_name="yolo26n",
        weights_file="/path/to/best.pt",
        output_dir="runs/football-video",
        tracker="botsort.yaml",
    )
)
print(outcome.outcome_markdown)
print(outcome.report_json)
```

Every run produces:

```text
outcome.md          # readable headline and per-class outcome
video_analysis.json # full aggregate report
frames.csv          # counts, confidence, and latency for every frame
detections.csv      # class, confidence, track ID, box, location, and occupancy
<annotated video>   # detections and track IDs rendered by Ultralytics
```

Video-only metrics include frame coverage, class distribution, confidence distribution
(min/max/mean/median/std/p5/p95/p99), detections per frame, normalized box occupancy, unique
tracks, sampled track length, preprocess/inference/postprocess latency, model pipeline FPS, and
end-to-end processing FPS.

> A plain video has no ground truth. Precision, recall, F1, AP/mAP, confusion-matrix accuracy,
> MOTA, MOTP, IDF1, and HOTA cannot be truthfully calculated from a video link alone. Label the
> frames and use the evaluator below for detection accuracy. Tracking accuracy additionally
> requires ground-truth object identities.

The weight file passed to video analysis must be a **fine-tuned detector checkpoint** such as
`best.pt`. `best_ssl.pt` contains training state and is not directly an object detector.

## Comprehensive labelled evaluation

```bash
ssldet evaluate \
  --model yolo26n \
  --weights /path/to/best.pt \
  --data /path/to/dataset.yaml \
  --split test \
  --output runs/football-evaluation
```

This retains every metric exposed by the installed Ultralytics validator: precision, recall,
F1, TP/FP/FN, mAP50, mAP75, mAP50-95, per-class AP, per-image metrics, fitness, validation
curves, confusion matrix, and preprocess/inference/postprocess timings. Task-specific box,
segmentation, pose, and OBB metrics are captured when the selected model exposes them. Reports
are written as `metrics.json`, `summary.csv`, `per_class_metrics.csv`,
`per_image_metrics.csv`, and `confusion_matrix.csv`; Ultralytics plots are kept in the same
directory.

```python
from ssldet import EvaluationConfig, evaluate

report = evaluate(
    EvaluationConfig(
        model_name="yolo26n",
        weights_file="/path/to/best.pt",
        data="/path/to/dataset.yaml",
        output_dir="runs/football-evaluation",
        split="test",
    )
)
print(report.metrics_json)
```

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

## Six methods, one interface

| Method | Learning signal | Two augmented views? | Moving target? | YOLO adaptation |
|---|---|---:|---:|---|
| SimCLR | NT-Xent contrastive loss | Yes | No | Global pooled backbone features |
| BYOL | Cross-view latent prediction | Yes | Yes | Online/EMA YOLO encoders |
| MoCo | Positive key + negative queue | Yes | Yes | Momentum YOLO key encoder |
| DINOv2 | Centered teacher/student cross-entropy + KoLeo | Multi-crop | Yes | EMA YOLO teacher |
| MAE | Masked pixel reconstruction | No | No | Lightweight CNN decoder on YOLO features |
| I-JEPA | Masked latent-block prediction | No | Yes | Transformer predictor over YOLO feature grid |

The MAE implementation is also a documented CNN-compatible adaptation rather than an exact copy
of the original ViT MAE architecture.

### Why the DINOv2 implementation is compute-scaled

Official DINOv2 trains ViT models with a large curated corpus and a combined DINO, iBOT, and
KoLeo recipe. This package provides a **YOLO-native compute-scaled DINOv2-style adaptation**:

1. Generate two global crops and configurable lower-resolution local crops.
2. Process every crop with the trainable student YOLO backbone.
3. Process only global crops with an exponential-moving-average teacher.
4. Match centered, sharpened teacher distributions across different student views.
5. Apply KoLeo nearest-neighbour entropy regularization to global features.
6. Transfer the trained student backbone directly into the YOLO detector.

It does not reproduce the official ViT patch-level iBOT objective or the LVD-142M training
regime. Reports should call it **YOLO-native compute-scaled DINOv2-style pretraining**, not an
official DINOv2 reproduction. See the
[official DINOv2 repository](https://github.com/facebookresearch/dinov2) and
[DINOv2 paper](https://arxiv.org/abs/2304.07193).

## Recommended dual-T4 Kaggle workflow

Create a YAML file based on `examples/ijepa_t4x2.yaml` or
`examples/dinov2_t4x2.yaml`, then launch one process per GPU:

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
ijepa_pretrained_yolo26n.pt # detector with the learned online backbone
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

detector = YOLO("/kaggle/working/ijepa_yolo26/ssl/ijepa_pretrained_yolo26n.pt")
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

# DINOv2-style
method: dinov2
local_crops: 4
local_crop_size: 96
dino_output_dim: 4096
student_temperature: 0.10
teacher_temperature: 0.04
center_momentum: 0.90
koleo_weight: 0.10

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

## License and third-party terms

The original `ssl-detection-lab` code is MIT-licensed. Dependencies and downloaded weights keep
their own terms; this project's MIT license does not override them.

- **Ultralytics:** its official documentation describes AGPL-3.0 for open-source use and an
  [Enterprise License](https://www.ultralytics.com/license) for use that does not follow the
  AGPL requirements. Review the [Ultralytics documentation](https://docs.ultralytics.com/)
  for your deployment. Research using YOLO26 should also use the
  [official Ultralytics YOLO26 citation](https://docs.ultralytics.com/models/yolo26/#citations-and-acknowledgments).
- **DINOv2:** standard DINOv2 code and backbone weights are Apache-2.0 according to the
  [official repository](https://github.com/facebookresearch/dinov2).
- **DINOv3:** code and weights use Meta's separate
  [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`CITATIONS.md`](CITATIONS.md). This
is a dependency notice, not legal advice; users remain responsible for checking the terms
applicable to their use and weights.
