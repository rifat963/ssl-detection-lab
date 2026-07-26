# Video Analysis

Streaming detection and tracking over a local file, URL, stream, or webcam — **without** labels.

## CLI

```bash
ssldet video \
  --model yolo26n \
  --weights runs/finetuned/best.pt \
  --source match.mp4 \
  --output runs/video \
  --tracker botsort.yaml \
  --stride 1
```

`--source` accepts a path, an HTTP/RTSP link, or a webcam index. Use `--tracker none` for
detection without tracking.

## Python

```python
from ssldet import VideoAnalysisConfig, analyze_video

result = analyze_video(VideoAnalysisConfig(
    video_source="match.mp4",
    model_name="yolo26n",
    weights_file="runs/finetuned/best.pt",
    output_dir="runs/video",
    tracker="botsort.yaml",
    max_frames=None,
).validate())
```

## Configuration

| Field | Default | Notes |
|---|---|---|
| `confidence` | 0.25 | Higher than evaluation — these are operational detections |
| `iou` | 0.7 | |
| `image_size` | 640 | |
| `tracker` | `"botsort.yaml"` | Or `"bytetrack.yaml"`, or `None` |
| `video_stride` | 1 | Analyse every Nth frame |
| `max_frames` | `None` | Stop early |
| `save_annotated` | `True` | Write the annotated video |
| `save_txt` / `save_confidence` | `False` | Per-frame label files |

Results stream frame by frame, so memory stays flat on long videos.

## Outputs

| File | Contents |
|---|---|
| `video_analysis.json` | Full report, schema-versioned |
| `outcome.md` | Human-readable summary with a per-class table |
| `frames.csv` | Per frame: detections, mean confidence, per-stage latency |
| `detections.csv` | Per detection: track ID, class, confidence, box, area ratio, centroid |

The report includes source metadata (fps, resolution, duration, frame count) probed via OpenCV
when the source is a local file, plus latency distributions (min/max/mean/median/std/p5/p95/p99),
throughput, per-class breakdowns, and track statistics.

## What you can and cannot conclude

This is **unlabelled** analysis. The report states this boundary explicitly in its
`metric_boundary` field, and `outcome.md` repeats it.

**Measurable without labels**

> detection counts · confidence distribution · box occupancy · latency · throughput ·
> frame coverage · class distribution · unique tracks · track length

**Requires ground truth — not available here**

> precision · recall · F1 · AP · mAP · confusion matrix · MOTA · MOTP · IDF1 · HOTA

A high mean confidence is not accuracy. A confident model can be confidently wrong. To make any
accuracy claim, use [Evaluation](Evaluation.md) on a labelled split.

## Two FPS numbers

The report distinguishes them; do not conflate them:

- **`processing_fps`** — end-to-end wall-clock throughput, including video decode and CSV
  accumulation. This is what your pipeline actually achieves.
- **`model_pipeline_fps`** — derived from preprocess + inference + postprocess time only. This is
  the model's speed in isolation and will be higher.

Quote `processing_fps` for deployment planning and `model_pipeline_fps` for model comparison.

## Tracking

Track IDs come from the Ultralytics tracker. `unique_tracks` counts distinct IDs, and
`track_length_sampled_frames` reports how long tracks persist **in sampled frames** — with
`video_stride > 1` that is not the same as source frames.

ID switches inflate `unique_tracks`. Without ground truth you cannot distinguish a genuine new
object from a switch, which is exactly why MOTA/IDF1/HOTA are listed as unavailable.

## Next

[Evaluation](Evaluation.md) · [Troubleshooting](Troubleshooting.md)
