# SSL Detection Lab

Pretrain Ultralytics YOLO backbones with self-supervised learning, transfer the result into an
object detector, evaluate it on labelled data, and analyse video — with no labels used during
pretraining.

Built for CSE445 Computer Vision labs and Kaggle T4 x2 runtimes. Heavy training loops live in
normal Python modules; notebooks hold configuration, plots, and interpretation.

## Where to start

| If you want to… | Go to |
|---|---|
| Run a lab in your browser, no setup | [Kaggle tutorials](../README.md#kaggle-tutorials--start-here) |
| Install locally or on a cluster | [Installation](Installation.md) |
| Pretrain something in 10 minutes | [Quickstart](Quickstart.md) |
| Pick an SSL objective | [SSL Methods](SSL-Methods.md) |
| Check whether your model is supported | [Model Families](Model-Families.md) |
| Move a checkpoint into a detector | [Downstream Transfer](Downstream-Transfer.md) |
| Score a detector on labelled data | [Evaluation](Evaluation.md) |
| Analyse a video or stream | [Video Analysis](Video-Analysis.md) |
| Fix an error | [Troubleshooting](Troubleshooting.md) |
| Add a method or send a patch | [Contributing](Contributing.md) |

## The one-paragraph version

```bash
pip install "git+https://github.com/rifat963/ssl-detection-lab.git@main"
ssldet doctor                 # verify torch / CUDA / ultralytics
ssldet models                 # list supported SSL objectives and model families
```

```python
from ssldet import PretrainConfig, pretrain

result = pretrain(PretrainConfig(
    method="simclr",
    image_roots=["/data/unlabelled/images"],
    output_dir="runs/simclr",
    yolo_model="yolo26n.yaml",
    epochs=25,
).validate())
print(result.yolo_checkpoint)
```

Then transfer the backbone into a detector and fine-tune it — see
[Downstream Transfer](Downstream-Transfer.md).

## Package layout

```text
ssldet.ssl          reusable objectives, losses, interfaces, custom-module registry
ssldet.backbones    YOLO, DINOv2, and DINOv3 feature encoders
ssldet.detection    backend-neutral detector protocol and Ultralytics adapter
ssldet.evaluation   labelled object-detection evaluation and metric export
ssldet.video        streaming detection/tracking and video metrics
ssldet.downstream   verified SSL-checkpoint transfer into YOLO detectors
ssldet.workflow     dry-run and distributed-launch helpers
```

## Scope and honesty

These are **compute-scaled, YOLO-native adaptations** of published methods, not reproductions of
the original papers' large-scale experiments. The MAE, DINOv2, and I-JEPA implementations are
documented adaptations to a CNN detector backbone. Describe them accordingly in reports — see
[SSL Methods](SSL-Methods.md) for what each one does and does not claim.

A falling SSL loss is **not** a detection metric. Only a labelled evaluation against a
matched baseline supports a claim about detection quality.

## Licensing

This code is MIT. Dependencies and downloaded weights keep their own terms — Ultralytics is
AGPL-3.0 with a commercial option, and DINOv3 weights use Meta's separate DINOv3 License. See
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) and [`CITATIONS.md`](../CITATIONS.md).
