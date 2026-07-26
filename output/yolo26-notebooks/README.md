# YOLO26 SSL tutorial notebooks

The Kaggle tutorial series for **YOLO26**, plus the all-in-one lab notebook. Each notebook is also
published publicly on Kaggle — see the table in the [project README](../../README.md#kaggle-labs)
for the direct links.

## Pretrain → downstream pairs

| Objective | SSL pretraining | Downstream transfer |
|---|---|---|
| SimCLR | [simclr_football_t4x2](simclr_football_t4x2_tutorial.ipynb) | [simclr_yolo26_football_downstream](simclr_yolo26_football_downstream_tutorial.ipynb) |
| BYOL | [byol_football_t4x2](byol_football_t4x2_tutorial.ipynb) | [byol_yolo26_football_downstream](byol_yolo26_football_downstream_tutorial.ipynb) |
| I-JEPA | [ijepa_yolo26_football_ssl](ijepa_yolo26_football_ssl_tutorial.ipynb) | [ijepa_yolo26_downstream](ijepa_yolo26_downstream_tutorial.ipynb) |
| DINOv3 | [dinov3_yolo26_football_ssl](dinov3_yolo26_football_ssl_tutorial.ipynb) | [dinov3_yolo26_downstream](dinov3_yolo26_downstream_tutorial.ipynb) |

Run the **SSL pretraining** notebook of a pair first — the downstream notebook consumes the
`best_ssl.pt` checkpoint it produces (or attach a previously saved checkpoint as a Kaggle input).

## All-in-one lab

[`ssl_detection_lab_football_t4x2_tutorial.ipynb`](ssl_detection_lab_football_t4x2_tutorial.ipynb)
audits the YOLO labels, dry-runs every self-contained SSL method, probes each supported model
family, fine-tunes one detector, exports labelled evaluation metrics, and exercises video
analysis.

## Requirements

- Kaggle **GPU T4 x2** with Internet enabled
- The `iasadpanwhar/football-player-detection-yolov8` dataset attached via **Add Input**
- `ssl-detection-lab` **0.8.1 or newer** — every notebook asserts this floor in its setup cell
- The DINOv3 notebooks additionally need PyTorch 2.7.1+ and user-supplied official Meta DINOv3
  weights, which carry their own separate DINOv3 License terms

## Naming: use `yolo26`, not `yolov26`

Ultralytics dropped the `v` from YOLO11 onward, so the architecture files are `yolo26n.yaml`,
`yolo26s.yaml`, and so on. A `yolov26*.yaml` name does not exist upstream and raises
`FileNotFoundError`.

For the same tutorials targeting a YOLO12 backbone, see [`../yolo12-notebooks/`](../yolo12-notebooks/).
