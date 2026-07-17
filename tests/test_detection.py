import ssldet.detection.ultralytics as ultralytics_backend
from ssldet.detection import (
    ULTRALYTICS_LICENSE_NOTICE,
    available_detection_backends,
    create_detector,
    load_detector,
)


class FakeModel:
    def predict(self, **kwargs):
        return "predict", kwargs

    def track(self, **kwargs):
        return "track", kwargs

    def val(self, **kwargs):
        return "validate", kwargs


def test_ultralytics_detector_delegates_common_operations(monkeypatch):
    monkeypatch.setattr(
        ultralytics_backend,
        "_load_ultralytics_model",
        lambda model_name, weights_file: FakeModel(),
    )
    detector = load_detector("yolo26n", "weights.pt")

    assert detector.predict(source="image.jpg")[0] == "predict"
    assert detector.track(source="video.mp4")[0] == "track"
    assert detector.validate(data="data.yaml")[0] == "validate"
    assert ULTRALYTICS_LICENSE_NOTICE["open_source_license"] == "AGPL-3.0"


def test_detection_factory_lists_and_creates_ultralytics_backend(monkeypatch):
    monkeypatch.setattr(
        ultralytics_backend,
        "_load_ultralytics_model",
        lambda model_name, weights_file: FakeModel(),
    )

    assert available_detection_backends() == ("ultralytics",)
    assert create_detector("yolo26n", "weights.pt").model_name == "yolo26n"
