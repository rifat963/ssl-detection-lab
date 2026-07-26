# Model Families

```bash
ssldet models            # human-readable
ssldet models --json     # machine-readable, for UIs and notebooks
```

```python
from ssldet import capabilities
capabilities()["model_families"]
```

## Support matrix

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
| RT-DETR | l, x | **No** | Yes | Yes |
| YOLO-NAS | s, m, l | **No** | Yes | Yes |

SSL pretraining requires an Ultralytics YAML-defined backbone whose final backbone node is a
single spatial `B x C x H x W` tensor. RT-DETR and YOLO-NAS do not expose one, so they are
evaluation- and video-only. Compatibility for custom models is checked at runtime.

## Naming: no `v` from YOLO11 onward

Ultralytics dropped the `v` starting with YOLO11. The architecture files are:

```text
yolo26n.yaml   yolo12n.yaml   yolo11n.yaml      correct
yolov26n.yaml  yolov12n.yaml  yolov11n.yaml     do not exist
```

Passing a non-existent name raises `FileNotFoundError` inside Ultralytics:

```python
YOLO("yolo12n.yaml")    # OK
YOLO("yolov12n.yaml")   # FileNotFoundError: 'yolov12n.yaml' does not exist
```

A **weights file** named `yolov12n.pt` will still load — the backend falls through to `YOLO(...)`
correctly — but `resolve_model_family()` does not recognize the spelling, so reports label it
*Custom Ultralytics YOLO* instead of *YOLO12*. Functionality is unaffected; only the reported
family name is wrong.

Older families genuinely are branded with a `v`, and both spellings resolve:

| Input | Resolves to |
|---|---|
| `yolov8s-seg`, `yolo8s` | YOLOv8 |
| `yolov10b`, `yolo10b` | YOLOv10 |
| `yolo-nas-s`, `yolonas-s` | YOLO-NAS |
| `rtdetr-l`, `rt-detr-l` | RT-DETR |

## `.yaml` vs `.pt`

This distinction decides what your experiment actually claims:

| Suffix | Initialization | Claim |
|---|---|---|
| `yolo26n.yaml` | Random | Strict label-free SSL pretraining |
| `yolo26n.pt` | COCO-supervised | Warm start + label-free domain adaptation |

`run_manifest.json` records which one you used in its `initialization` field. Quote it in reports
rather than restating it from memory.

## Scaling

Change the letter to change capacity: `yolo26n` → `yolo26s` → `yolo26m` → `yolo26l` → `yolo26x`.

The SSL checkpoint is tied to the scale it was trained on. Transferring a `yolo26n` encoder into
a `yolo26s` detector fails the coverage check — see
[Downstream Transfer](Downstream-Transfer.md#coverage).

## Detection backends

```python
from ssldet.detection import available_detection_backends, load_detector

available_detection_backends()          # ('ultralytics',)
detector = load_detector("yolo26n", "yolo26n.pt")
detector.predict(source="image.jpg", conf=0.25)
detector.validate(data="dataset.yaml", split="test")
detector.track(source="video.mp4", tracker="botsort.yaml", stream=True)
```

The backend dispatches to `RTDETR`, `NAS`, or `YOLO` based on the resolved family. Register your
own with `register_detection_backend()`.

## Next

[SSL Methods](SSL-Methods.md) · [Evaluation](Evaluation.md)
