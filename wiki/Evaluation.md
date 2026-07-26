# Evaluation

Scoring a detector against **labelled** data, exporting every metric Ultralytics exposes.

## CLI

```bash
ssldet evaluate \
  --model yolo26n \
  --weights runs/finetuned/best.pt \
  --data dataset.yaml \
  --output runs/eval \
  --split test \
  --imgsz 640 --batch 16
```

## Python

```python
from ssldet import EvaluationConfig, evaluate

result = evaluate(EvaluationConfig(
    model_name="yolo26n",
    weights_file="runs/finetuned/best.pt",
    data="dataset.yaml",
    output_dir="runs/eval",
    split="test",
).validate())
```

## Configuration

| Field | Default | Notes |
|---|---|---|
| `split` | `"val"` | `train`, `val`, or `test` |
| `image_size` | 640 | ≥ 32 |
| `batch_size` | 16 | |
| `confidence` | 0.001 | Low by design — mAP integrates the full PR curve |
| `iou` | 0.7 | NMS threshold |
| `max_detections` | 300 | |
| `device` | `None` | `"cpu"`, `"0"`, `"0,1"`, … |
| `half` | `False` | FP16 inference |
| `plots` | `True` | Ultralytics curve images |
| `save_json` | `True` | COCO-format predictions |

> Keep `confidence` at 0.001 for mAP. Raising it truncates the precision-recall curve and inflates
> precision while deflating recall — the resulting number is not comparable to published mAP.

## Outputs

| File | Contents |
|---|---|
| `metrics.json` | Everything, schema-versioned |
| `summary.csv` | Headline table |
| `per_class_metrics.csv` | Precision, recall, F1, AP50, AP50-95 per class |
| `per_image_metrics.csv` | Per-image precision/recall/F1/TP/FP/FN where exposed |
| `confusion_matrix.csv` | Labelled matrix, rows = predictions, columns = ground truth |

`metrics.json` holds the raw `results_dict`, task aggregates (`box`, `seg`, `pose`, `obb`,
`probs`), speed, fitness, curves, per-class and per-image rows, and class names.

Non-finite floats become `null` so the JSON is always valid and parseable — no `NaN` tokens.

## Confusion matrix

Ultralytics orders rows as predictions and columns as ground truth. The exporter labels both axes
with real class names and appends a `background` row/column for detection tasks. Duplicate class
names are de-duplicated (`car`, `car_1`) so the CSV header stays unique and machine-readable.

## Interpreting results

Metrics that need ground truth — available here:

> precision · recall · F1 · AP · mAP50 · mAP75 · mAP50-95 · confusion matrix

For a label-efficiency claim, report **mean ± standard deviation across at least three seeds**,
and hold every downstream setting constant across conditions. A single run is an anecdote.

Compare against the four conditions in
[the suggested protocol](../README.md#suggested-experimental-protocol): random init, COCO
pretrained, strict SSL, and COCO + SSL adaptation.

## Next

[Video Analysis](Video-Analysis.md) · [Downstream Transfer](Downstream-Transfer.md)
