import json
from types import SimpleNamespace

import ssldet.evaluation as evaluation_module
import ssldet.video as video_module
from ssldet.evaluation import EvaluationConfig, evaluate
from ssldet.video import VideoAnalysisConfig, analyze_video


class _MetricComponent:
    mp, mr, map, map50, map75 = 0.8, 0.7, 0.65, 0.9, 0.75
    maps = [0.6, 0.7]
    ap_class_index = [0, 1]
    image_metrics = {
        "frame1.jpg": {
            "precision": 1.0,
            "recall": 0.5,
            "f1": 2 / 3,
            "tp": 1,
            "fp": 0,
            "fn": 1,
        }
    }

    def class_result(self, index):
        return 0.8, 0.7, 0.9, 0.65


class _FakeMetrics:
    names = {0: "player", 1: "ball"}
    box = _MetricComponent()
    seg = pose = obb = probs = None
    results_dict = {"metrics/precision(B)": 0.8, "metrics/mAP50-95(B)": 0.65}
    speed = {"preprocess": 1.0, "inference": 5.0, "postprocess": 2.0}
    fitness = 0.65
    curves_results = []
    confusion_matrix = SimpleNamespace(matrix=[[5, 1, 0], [1, 3, 0], [0, 1, 0]])
    task = "detect"

    def summary(self):
        return [{"Class": "all", "Images": 1, "Instances": 2, "Box-P": 0.8}]


class _FakeEvaluationModel:
    def val(self, **kwargs):
        return _FakeMetrics()


class _FakeVideoModel:
    def track(self, **kwargs):
        for index in range(3):
            boxes = SimpleNamespace(
                conf=[0.8, 0.6],
                cls=[0, 1],
                id=[1, index + 2],
                xyxy=[[10, 20, 50, 80], [100, 100, 180, 220]],
            )
            yield SimpleNamespace(
                boxes=boxes,
                obb=None,
                names={0: "player", 1: "ball"},
                task="detect",
                orig_shape=(240, 320),
                speed={"preprocess": 1.0, "inference": 5.0, "postprocess": 2.0},
            )


def test_evaluation_exports_correct_per_class_metric_names(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluation_module, "_load_model", lambda model_name, weights: _FakeEvaluationModel()
    )
    result = evaluate(EvaluationConfig("yolo26n", "best.pt", "data.yaml", str(tmp_path)))
    report = json.loads(result.metrics_json.read_text(encoding="utf-8"))

    assert report["per_class"][0]["ap50"] == 0.9
    assert report["per_class"][0]["ap50_95"] == 0.65
    assert report["per_class"][0]["f1"] == 2 * 0.8 * 0.7 / 1.5
    assert report["per_image"][0]["image"] == "frame1.jpg"
    assert result.confusion_matrix_csv is not None


def test_video_analysis_exports_aggregate_and_tabular_reports(monkeypatch, tmp_path):
    monkeypatch.setattr(
        video_module, "_load_model", lambda model_name, weights: _FakeVideoModel()
    )
    result = analyze_video(
        VideoAnalysisConfig(
            "https://example.com/video.mp4",
            "yolo26n",
            "best.pt",
            str(tmp_path),
            save_annotated=False,
        )
    )
    report = json.loads(result.report_json.read_text(encoding="utf-8"))

    assert report["video_metrics"]["frames_processed"] == 3
    assert report["video_metrics"]["total_detections"] == 6
    assert report["video_metrics"]["tracking"]["unique_tracks"] == 4
    assert result.frames_csv.exists()
    assert result.detections_csv.exists()
    assert result.outcome_markdown.exists()

