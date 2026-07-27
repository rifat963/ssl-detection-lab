# Tracker notebooks

Multi-object tracking and **data association** — the step that turns per-frame detections into
object identities that persist across frames.

| Notebook | What it teaches |
|---|---|
| [tracker_data_association_tutorial](tracker_data_association_tutorial.ipynb) | The full tracker lab: catalog, six-tracker comparison, lifetime analysis, config tuning, and the metric boundary |

## What the lab does

1. Reads the tracker catalog (`capabilities()["trackers"]`) — six trackers, their association
   strategy, ReID and motion-compensation support
2. Runs BoT-SORT over a video and reads back the **resolved** tracker settings from
   `video_analysis.json`
3. Measures track lifetimes and continuity from `detections.csv`
4. Runs **all six trackers** over identical frames with identical detections, so every
   difference is attributable to association alone
5. Sweeps `track_buffer` with a custom tracker YAML
6. Separates what the results support from what they do not

## Requirements

- `ssl-detection-lab` **0.9.0 or newer** — the notebook asserts this floor. Earlier versions have
  no tracker catalog, no tracker validation, and do not record tracker settings in the report.
- A **fine-tuned detector** checkpoint (`best.pt` from any downstream lab). `best_ssl.pt` is SSL
  training state, not a detector, and will not work. The notebook falls back to COCO-pretrained
  `yolo26n.pt` so the pipeline still runs.
- A video. The notebook probes the Kaggle football dataset paths first and tells you exactly what
  it tried if nothing is found.
- GPU is optional. Tracking is inference-only; lower `MAX_FRAMES` to run on CPU.

## The one thing to take away

Trackers are **not trained** by this package and are unaffected by SSL pretraining. They consume
whatever the detector emits. SSL improves tracking only by improving detections.

And because a plain video has no ground-truth identities, **no accuracy claim is available here**
— not MOTA, not IDF1, not HOTA, not ID-switch counts. Fewer unique tracks is not "better"; it is
equally consistent with two objects being merged into one ID. The notebook returns to this point
deliberately, because it is the easiest mistake to make when writing up tracking results.

For labelled MOT evaluation, use an annotated sequence (MOT17, MOT20, DanceTrack) with a metrics
library such as [TrackEval](https://github.com/JonathonLuiten/TrackEval).

## Tracker configuration files

The six tracker YAMLs ship with Ultralytics under **AGPL-3.0** and are deliberately **not**
vendored into this MIT-licensed repository. Copy one out of your install to customize it:

```python
import shutil
from ultralytics.utils.checks import check_yaml

shutil.copy(check_yaml("botsort.yaml"), "my_botsort.yaml")
```

Reference: [wiki/Video-Analysis](../../wiki/Video-Analysis.md#supported-trackers) ·
[Ultralytics tracking docs](https://docs.ultralytics.com/modes/track/)
