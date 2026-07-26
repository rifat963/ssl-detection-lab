"""Backend-adapted object detection, validation, and tracking."""

from ssldet.detection import load_detector

detector = load_detector("yolo26n", "yolo26n.pt")
predictions = detector.predict(source="image.jpg", conf=0.25)
validation = detector.validate(data="dataset.yaml", split="test")
tracks = detector.track(
    source="video.mp4",
    tracker="botsort.yaml",
    stream=True,
)

print(predictions, validation, tracks)
