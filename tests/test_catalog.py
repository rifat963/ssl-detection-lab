from ssldet.catalog import SSL_ARCHITECTURES, capabilities, resolve_model_family


def test_catalog_lists_every_ssl_architecture():
    assert {item.name for item in SSL_ARCHITECTURES} == {
        "simclr",
        "byol",
        "moco",
        "dinov2",
        "mae",
        "ijepa",
    }
    assert capabilities()["model_families"]
    assert len(capabilities()["dinov2_feature_backbones"]) == 8


def test_model_family_resolution_handles_variants_and_custom_weights():
    assert resolve_model_family("yolo26n").name == "YOLO26"
    assert resolve_model_family("yolo12s").name == "YOLO12"
    assert resolve_model_family("YOLOv8x-seg").name == "YOLOv8"
    assert resolve_model_family("rt-detr-l").name == "RT-DETR"
    assert resolve_model_family("my-football-detector").name == "Custom Ultralytics YOLO"
