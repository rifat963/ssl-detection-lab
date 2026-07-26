from pathlib import Path

import pytest

from ssldet.evaluation import EvaluationConfig
from ssldet.video import VideoAnalysisConfig, _stats


def test_evaluation_config_rejects_invalid_thresholds():
    config = EvaluationConfig("yolo26n", "best.pt", "data.yaml", confidence=1.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        config.validate()


def test_video_config_accepts_links_and_limits():
    config = VideoAnalysisConfig(
        "https://example.com/video.mp4", "yolo26n", "best.pt", max_frames=10
    )
    assert config.validate().max_frames == 10


def test_video_config_accepts_webcam_indices_and_path_objects():
    assert VideoAnalysisConfig(0, "yolo26n", Path("best.pt")).validate().video_source == 0
    assert EvaluationConfig(
        "yolo26n", Path("best.pt"), Path("data.yaml")
    ).validate().weights_file == Path("best.pt")


def test_stats_exports_tail_percentiles():
    values = [float(value) for value in range(1, 101)]
    summary = _stats(values)
    assert summary["mean"] == 50.5
    assert summary["p95"] == pytest.approx(95.05)
    assert summary["p99"] == pytest.approx(99.01)
