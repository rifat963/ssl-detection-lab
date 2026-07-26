import json
from pathlib import Path
from types import SimpleNamespace

import ssldet.evaluation as evaluation_module
import ssldet.video as video_module
from ssldet.evaluation import EvaluationConfig, evaluate
from ssldet.video import VideoAnalysisConfig, analyze_video


class _VideoModel:
    def predict(self, **kwargs):
        for _ in range(2):
            yield SimpleNamespace(
                boxes=SimpleNamespace(
                    conf=[0.9], cls=[0], id=None, xyxy=[[0, 0, 10, 10]]
                ),
                obb=None,
                names={0: "player"},
                task="detect",
                orig_shape=(120, 160),
                speed={"preprocess": 1.0, "inference": 4.0, "postprocess": 1.0},
            )


def test_video_report_serializes_path_weights(monkeypatch, tmp_path):
    """weights_file is declared as ``str | Path``; a Path must not break the JSON export.

    The report is written only after the whole video has been processed, so an
    unserializable value discards all completed inference work.
    """

    monkeypatch.setattr(video_module, "_load_model", lambda model_name, weights: _VideoModel())
    result = analyze_video(
        VideoAnalysisConfig(
            video_source=Path("clip.mp4"),
            model_name="yolo26n",
            weights_file=Path("weights") / "best.pt",
            output_dir=str(tmp_path),
            tracker=None,
            save_annotated=False,
        )
    )
    report = json.loads(result.report_json.read_text(encoding="utf-8"))
    assert report["model"]["weights"] == str(Path("weights") / "best.pt")
    assert report["video_metrics"]["frames_processed"] == 2


class _NonMappingImageMetrics:
    ap_class_index = []
    # A build that reports a bare score per image rather than a metric mapping.
    image_metrics = {"frame1.jpg": [0.5, 0.6]}


class _Metrics:
    names = {0: "player"}
    box = _NonMappingImageMetrics()
    seg = pose = obb = probs = None
    results_dict = {"metrics/mAP50-95(B)": 0.4}
    speed = {"inference": 3.0}
    fitness = 0.4
    curves_results = []
    confusion_matrix = None
    task = "detect"

    def summary(self):
        return [{"Class": "all", "Box-P": 0.5}]


def test_evaluation_tolerates_non_mapping_image_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluation_module, "_load_model", lambda model_name, weights: SimpleNamespace(
            val=lambda **kwargs: _Metrics()
        )
    )
    result = evaluate(EvaluationConfig("yolo26n", "best.pt", "data.yaml", str(tmp_path)))
    report = json.loads(result.metrics_json.read_text(encoding="utf-8"))
    assert report["per_image"][0]["image"] == "frame1.jpg"
    assert report["per_image"][0]["value"] == [0.5, 0.6]
