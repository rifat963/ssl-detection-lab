# Downstream Transfer

Moving a pretrained SSL encoder into a real detector.

## What transfers, and what does not

`transfer_ssl_backbone_to_yolo` copies **only** the trainable online/student encoder. Everything
that exists purely to make SSL work is discarded:

| Transferred | Discarded |
|---|---|
| `online_encoder.*` / `student_encoder.*` | Projection head, predictor |
| | Target / teacher network |
| | Optimizer, scheduler, gradient scaler |
| | Training history |

That list is also written into the transfer report as `excluded_ssl_components`.

## Usage

```python
from ssldet import transfer_ssl_backbone_to_yolo

transfer = transfer_ssl_backbone_to_yolo(
    "runs/simclr/best_ssl.pt",
    "runs/simclr/backbone.pt",
    yolo_model="yolo26n.yaml",     # optional if the checkpoint records it
    minimum_coverage=0.95,
)

print(transfer.coverage)            # 1.0
print(transfer.loaded_keys, "/", transfer.total_backbone_keys)
print(transfer.report_json)         # backbone.transfer.json
```

`yolo_model` is optional — the SSL checkpoint stores `config.yolo_model` and it is used
automatically. Pass it explicitly only to override.

## Coverage

Coverage is the fraction of the target backbone's state-dict keys the checkpoint supplies.

**Expect 100%.** Anything lower means a mismatch, and the call raises below `minimum_coverage`
rather than silently producing a half-initialized detector.

Common causes of low coverage:

| Symptom | Cause |
|---|---|
| Coverage near 0 | Wrong model family (e.g. YOLO26 encoder into YOLO12) |
| Coverage well below 1 | Same family, different scale (`n` encoder into `s` detector) |
| `RuntimeError` about tensor shapes | Same key names, incompatible shapes — again a scale mismatch |

The rule: **transfer into the same family and scale you pretrained on.**

## The output

The result is an ordinary Ultralytics checkpoint. Fine-tune it normally:

```python
from ultralytics import YOLO

YOLO("runs/simclr/backbone.pt").train(data="dataset.yaml", epochs=50, imgsz=640)
```

A sibling `*.transfer.json` records the SSL method, source model, encoder prefix, key counts,
coverage, and any missing/unexpected keys — keep it with your results for provenance.

## Which checkpoint to transfer

Use `best_ssl.pt` or `last_ssl.pt` — the **full SSL checkpoint**.

Do *not* fine-tune `<method>_pretrained_<model>.pt` expecting it to behave differently: that file
is already a detector-compatible model saved directly from the trained backbone. Both routes give
you the same learned weights; the transfer helper additionally verifies coverage and writes a
provenance report, which is why it is the recommended path.

## Encoder prefix

The helper auto-detects `online_encoder.` or `student_encoder.` (DINOv2 uses the latter), and
strips a `module.` prefix left by DDP. Override for a custom objective:

```python
transfer_ssl_backbone_to_yolo(..., encoder_prefix="my_encoder.")
```

## Fine-tuning advice

1. **Warm up the head first.** The detection head is randomly initialized; a short head-only
   warmup before unfreezing the backbone avoids destroying the pretrained features.
2. **Keep settings identical across conditions.** Otherwise you are comparing schedules, not
   pretraining.
3. **Always train a from-scratch baseline.** SSL numbers without a matched baseline say nothing.

See [the suggested experimental protocol](../README.md#suggested-experimental-protocol) for a
defensible four-condition comparison.

## Next

[Evaluation](Evaluation.md) · [Model Families](Model-Families.md)
