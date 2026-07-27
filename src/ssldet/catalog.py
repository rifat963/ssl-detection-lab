"""Supported self-supervised methods and downstream model families.

The catalog is deliberately data-only so applications can render it before loading
PyTorch or Ultralytics models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SSLArchitecture:
    name: str
    category: str
    objective: str
    views: int
    moving_target: bool
    video_ready_after_finetuning: bool = True


@dataclass(frozen=True)
class ModelFamily:
    name: str
    aliases: tuple[str, ...]
    scales: tuple[str, ...]
    ssl_backbone: bool
    dataset_evaluation: bool
    video_analysis: bool
    tasks: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class FeatureBackbone:
    name: str
    architecture: str
    patch_size: int
    embedding_dim: int
    registers: bool
    object_detector: bool = False
    pretrained_weights: bool = True
    license_name: str = ""


@dataclass(frozen=True)
class DetectionBackend:
    name: str
    model_families: tuple[str, ...]
    tasks: tuple[str, ...]
    open_source_license: str
    commercial_license_available: bool
    license_url: str


@dataclass(frozen=True)
class Tracker:
    """One multi-object tracker shipped by Ultralytics for ``mode="track"``."""

    name: str
    config_file: str
    association: str
    appearance_reid: bool
    motion_compensation: bool
    paper_url: str
    pick_it_when: str = ""


SSL_ARCHITECTURES = (
    SSLArchitecture("simclr", "contrastive", "NT-Xent cross-view agreement", 2, False),
    SSLArchitecture("byol", "non-contrastive", "online-to-EMA latent prediction", 2, True),
    SSLArchitecture("moco", "contrastive", "momentum encoder with negative queue", 2, True),
    SSLArchitecture(
        "dinov2",
        "self-distillation",
        "multi-crop teacher/student distillation with centering and KoLeo",
        2,
        True,
    ),
    SSLArchitecture(
        "dinov3",
        "foundation-model distillation",
        "frozen DINOv3 global and dense feature regression into a YOLO student",
        1,
        False,
    ),
    SSLArchitecture("mae", "generative", "masked pixel reconstruction", 1, False),
    SSLArchitecture("ijepa", "predictive", "masked latent-block prediction", 1, True),
)


DINOV2_FEATURE_BACKBONES = tuple(
    FeatureBackbone(
        f"dinov2_{architecture}{'_reg' if registers else ''}",
        f"ViT-{scale}/14",
        14,
        embedding_dim,
        registers,
        license_name="Apache-2.0",
    )
    for architecture, scale, embedding_dim in (
        ("vits14", "S", 384),
        ("vitb14", "B", 768),
        ("vitl14", "L", 1024),
        ("vitg14", "g", 1536),
    )
    for registers in (False, True)
)


DINOV3_FEATURE_BACKBONES = tuple(
    FeatureBackbone(
        name,
        architecture,
        patch_size,
        embedding_dim,
        registers,
        pretrained_weights=True,
        license_name="DINOv3 License",
    )
    for name, architecture, patch_size, embedding_dim, registers in (
        ("dinov3_vits16", "ViT-S/16", 16, 384, True),
        ("dinov3_vits16plus", "ViT-S+/16", 16, 384, True),
        ("dinov3_vitb16", "ViT-B/16", 16, 768, True),
        ("dinov3_vitl16", "ViT-L/16", 16, 1024, True),
        ("dinov3_vith16plus", "ViT-H+/16", 16, 1280, True),
        ("dinov3_vit7b16", "ViT-7B/16", 16, 4096, True),
        ("dinov3_convnext_tiny", "ConvNeXt Tiny", 32, 768, False),
        ("dinov3_convnext_small", "ConvNeXt Small", 32, 768, False),
        ("dinov3_convnext_base", "ConvNeXt Base", 32, 1024, False),
        ("dinov3_convnext_large", "ConvNeXt Large", 32, 1536, False),
    )
)


DETECTION_BACKENDS = (
    DetectionBackend(
        "Ultralytics",
        (
            "YOLO26",
            "YOLO12",
            "YOLO11",
            "YOLOv10",
            "YOLOv9",
            "YOLOv8",
            "YOLOv6",
            "YOLOv5u",
            "YOLOv3u",
            "RT-DETR",
            "YOLO-NAS",
            "Custom Ultralytics YOLO",
        ),
        ("detect", "segment", "semantic", "pose", "obb", "classify", "track"),
        "AGPL-3.0",
        True,
        "https://www.ultralytics.com/license",
    ),
)


# Trackers consume detections; they are not trained by this package and are unaffected by
# SSL pretraining. The configuration files themselves ship with Ultralytics under AGPL-3.0
# and are deliberately not vendored here -- they are resolved from the install at runtime.
TRACKERS = (
    Tracker(
        "botsort",
        "botsort.yaml",
        "IoU + optional ReID, with global motion compensation",
        True,
        True,
        "https://arxiv.org/abs/2206.14651",
        "Default. Moving camera or frequent occlusion.",
    ),
    Tracker(
        "bytetrack",
        "bytetrack.yaml",
        "IoU over high- and low-confidence detections",
        False,
        False,
        "https://arxiv.org/abs/2110.06864",
        "Fastest and simplest; static camera, clear separation.",
    ),
    Tracker(
        "ocsort",
        "ocsort.yaml",
        "Observation-centric IoU with velocity consistency",
        False,
        False,
        "https://arxiv.org/abs/2203.14360",
        "Nonlinear motion and short occlusions without ReID cost.",
    ),
    Tracker(
        "deepocsort",
        "deepocsort.yaml",
        "Observation-centric IoU plus appearance embeddings",
        True,
        True,
        "https://arxiv.org/abs/2302.11813",
        "Crowded scenes where identity switches dominate the error.",
    ),
    Tracker(
        "fasttrack",
        "fasttrack.yaml",
        "ByteTrack-style IoU with occlusion-aware Kalman rollback",
        False,
        False,
        "https://arxiv.org/abs/2508.14370",
        "Heavy mutual occlusion at ByteTrack-like cost.",
    ),
    Tracker(
        "tracktrack",
        "tracktrack.yaml",
        "Multi-cue HMIoU, confidence, and angle with iterative assignment",
        True,
        True,
        "https://openaccess.thecvf.com/content/CVPR2025/html/"
        "Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html",
        "Best association quality when the extra compute is acceptable.",
    ),
)


# SSL compatibility means the model exposes an Ultralytics YAML backbone whose final
# backbone node is a spatial BxCxHxW tensor. Evaluation is broader than SSL pretraining.
MODEL_FAMILIES = (
    ModelFamily(
        "YOLO26",
        ("yolo26",),
        ("n", "s", "m", "l", "x"),
        True,
        True,
        True,
        ("detect", "segment", "semantic", "pose", "obb", "classify"),
        "Current Ultralytics family; P2/P6 YAML variants can also be used.",
    ),
    ModelFamily(
        "YOLO12",
        ("yolo12",),
        ("n", "s", "m", "l", "x"),
        True,
        True,
        True,
        ("detect", "segment", "pose", "obb", "classify"),
        "Attention-centric research family; only detection has official pretrained weights.",
    ),
    ModelFamily(
        "YOLO11",
        ("yolo11",),
        ("n", "s", "m", "l", "x"),
        True,
        True,
        True,
        ("detect", "segment", "pose", "obb", "classify"),
    ),
    ModelFamily(
        "YOLOv10",
        ("yolov10", "yolo10"),
        ("n", "s", "m", "b", "l", "x"),
        True,
        True,
        True,
        ("detect",),
    ),
    ModelFamily(
        "YOLOv9",
        ("yolov9", "yolo9"),
        ("t", "s", "m", "c", "e"),
        True,
        True,
        True,
        ("detect", "segment"),
    ),
    ModelFamily(
        "YOLOv8",
        ("yolov8", "yolo8"),
        ("n", "s", "m", "l", "x"),
        True,
        True,
        True,
        ("detect", "segment", "pose", "obb", "classify"),
    ),
    ModelFamily(
        "YOLOv6",
        ("yolov6", "yolo6"),
        ("n", "s", "m", "l", "x"),
        True,
        True,
        True,
        ("detect",),
        "Ultralytics provides native YAML architectures; train weights before evaluation.",
    ),
    ModelFamily(
        "YOLOv5u",
        ("yolov5u", "yolov5", "yolo5"),
        ("n", "s", "m", "l", "x"),
        True,
        True,
        True,
        ("detect", "segment", "classify"),
        "Use Ultralytics-compatible v5u checkpoints, not legacy repository checkpoints.",
    ),
    ModelFamily(
        "YOLOv3u",
        ("yolov3u", "yolov3", "yolo3"),
        ("standard", "spp", "tiny"),
        True,
        True,
        True,
        ("detect",),
        "Use current Ultralytics-compatible v3u/YAML models.",
    ),
    ModelFamily(
        "RT-DETR",
        ("rtdetr", "rt-detr"),
        ("l", "x"),
        False,
        True,
        True,
        ("detect",),
        "Evaluation/video inference only; not supported by the YOLO SSL backbone adapter.",
    ),
    ModelFamily(
        "YOLO-NAS",
        ("yolo-nas", "yolonas"),
        ("s", "m", "l"),
        False,
        True,
        True,
        ("detect",),
        "Evaluation/video inference only; not supported by the YOLO SSL backbone adapter.",
    ),
    ModelFamily(
        "Custom Ultralytics YOLO",
        ("custom",),
        (),
        True,
        True,
        True,
        ("detect", "segment", "pose", "obb", "classify"),
        "SSL requires a YAML-defined spatial backbone; compatibility is checked at runtime.",
    ),
)


def resolve_model_family(model_name: str) -> ModelFamily:
    """Resolve a user-facing model name to a catalog family."""

    normalized = model_name.lower().replace("_", "-").replace(" ", "").strip()
    for family in MODEL_FAMILIES:
        for alias in family.aliases:
            compact_alias = alias.replace("_", "-").replace(" ", "")
            if normalized.startswith(compact_alias):
                return family
    return MODEL_FAMILIES[-1]


def available_tracker_names() -> tuple[str, ...]:
    """Return the built-in tracker names in catalog order."""

    return tuple(tracker.name for tracker in TRACKERS)


def resolve_tracker(tracker: str) -> Tracker:
    """Resolve a tracker name or bundled config filename to its catalog entry.

    Accepts ``"botsort"`` and ``"botsort.yaml"`` alike. Raises ``ValueError`` for an
    unknown name so a typo fails during configuration rather than part-way through a
    video. Custom local tracker files are handled by the caller, not here.
    """

    normalized = Path(str(tracker)).stem.lower().replace("-", "").replace("_", "").strip()
    for item in TRACKERS:
        if item.name == normalized:
            return item
    raise ValueError(
        f"Unknown tracker {tracker!r}; choose from {list(available_tracker_names())}, "
        "or pass the path to an existing custom tracker YAML."
    )


def capabilities() -> dict[str, list[dict]]:
    """Return a JSON-serializable catalog for CLIs, notebooks, and web UIs."""

    return {
        "ssl_architectures": [asdict(item) for item in SSL_ARCHITECTURES],
        "model_families": [asdict(item) for item in MODEL_FAMILIES],
        "dinov2_feature_backbones": [asdict(item) for item in DINOV2_FEATURE_BACKBONES],
        "dinov3_feature_backbones": [asdict(item) for item in DINOV3_FEATURE_BACKBONES],
        "object_detection_backends": [asdict(item) for item in DETECTION_BACKENDS],
        "trackers": [asdict(item) for item in TRACKERS],
    }
