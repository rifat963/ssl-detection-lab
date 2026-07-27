"""SSL pretraining, detector evaluation, and video analysis building blocks.

Heavy training/inference modules are imported lazily so the model catalog can be
displayed even before PyTorch and Ultralytics are installed.
"""

from .catalog import (
    DETECTION_BACKENDS,
    DINOV2_FEATURE_BACKBONES,
    DINOV3_FEATURE_BACKBONES,
    MODEL_FAMILIES,
    SSL_ARCHITECTURES,
    TRACKERS,
    available_tracker_names,
    capabilities,
    resolve_tracker,
)

__all__ = [
    "DETECTION_BACKENDS",
    "DINOV2_FEATURE_BACKBONES",
    "DINOV3_FEATURE_BACKBONES",
    "MODEL_FAMILIES",
    "SSL_ARCHITECTURES",
    "TRACKERS",
    "available_tracker_names",
    "resolve_tracker",
    "EvaluationConfig",
    "EvaluationResult",
    "DistributedPretrainResult",
    "DINOv3YOLOConfig",
    "DINOv3YOLOResult",
    "BackboneTransferResult",
    "PretrainConfig",
    "PretrainResult",
    "VideoAnalysisConfig",
    "VideoAnalysisResult",
    "analyze_video",
    "assert_supported_runtime",
    "available_detection_backends",
    "available_ssl_modules",
    "build_dinov2_transform",
    "build_dinov3_transform",
    "capabilities",
    "create_detector",
    "create_ssl_module",
    "evaluate",
    "launch_distributed_pretrain",
    "load_dinov2_backbone",
    "load_dinov3_backbone",
    "load_detector",
    "make_dry_run_config",
    "pretrain",
    "pretrain_dinov3_yolo26",
    "runtime_report",
    "transfer_ssl_backbone_to_yolo",
    "validate_dinov3_yolo26_support",
]
__version__ = "0.9.0"


def __getattr__(name: str):
    if name in {
        "DINOv3YOLOConfig",
        "DINOv3YOLOResult",
        "pretrain_dinov3_yolo26",
        "validate_dinov3_yolo26_support",
    }:
        from .dinov3 import (
            DINOv3YOLOConfig,
            DINOv3YOLOResult,
            pretrain_dinov3_yolo26,
            validate_dinov3_yolo26_support,
        )

        return {
            "DINOv3YOLOConfig": DINOv3YOLOConfig,
            "DINOv3YOLOResult": DINOv3YOLOResult,
            "pretrain_dinov3_yolo26": pretrain_dinov3_yolo26,
            "validate_dinov3_yolo26_support": validate_dinov3_yolo26_support,
        }[name]
    if name in {"BackboneTransferResult", "transfer_ssl_backbone_to_yolo"}:
        from .downstream import BackboneTransferResult, transfer_ssl_backbone_to_yolo

        return {
            "BackboneTransferResult": BackboneTransferResult,
            "transfer_ssl_backbone_to_yolo": transfer_ssl_backbone_to_yolo,
        }[name]
    if name in {"available_ssl_modules", "create_ssl_module"}:
        from .ssl import available_ssl_modules, create_ssl_module

        return {
            "available_ssl_modules": available_ssl_modules,
            "create_ssl_module": create_ssl_module,
        }[name]
    if name in {"available_detection_backends", "create_detector", "load_detector"}:
        from .detection import available_detection_backends, create_detector, load_detector

        return {
            "available_detection_backends": available_detection_backends,
            "create_detector": create_detector,
            "load_detector": load_detector,
        }[name]
    if name in {"assert_supported_runtime", "runtime_report"}:
        from .runtime import assert_supported_runtime, runtime_report

        return {
            "assert_supported_runtime": assert_supported_runtime,
            "runtime_report": runtime_report,
        }[name]
    if name in {
        "DistributedPretrainResult",
        "launch_distributed_pretrain",
        "make_dry_run_config",
    }:
        from .workflow import (
            DistributedPretrainResult,
            launch_distributed_pretrain,
            make_dry_run_config,
        )

        return {
            "DistributedPretrainResult": DistributedPretrainResult,
            "launch_distributed_pretrain": launch_distributed_pretrain,
            "make_dry_run_config": make_dry_run_config,
        }[name]
    if name in {"build_dinov2_transform", "load_dinov2_backbone"}:
        from .backbones import build_dinov2_transform, load_dinov2_backbone

        return {
            "build_dinov2_transform": build_dinov2_transform,
            "load_dinov2_backbone": load_dinov2_backbone,
        }[name]
    if name in {"build_dinov3_transform", "load_dinov3_backbone"}:
        from .backbones import build_dinov3_transform, load_dinov3_backbone

        return {
            "build_dinov3_transform": build_dinov3_transform,
            "load_dinov3_backbone": load_dinov3_backbone,
        }[name]
    if name == "PretrainConfig":
        from .config import PretrainConfig

        return PretrainConfig
    if name in {"PretrainResult", "pretrain"}:
        from .trainer import PretrainResult, pretrain

        return {"PretrainResult": PretrainResult, "pretrain": pretrain}[name]
    if name in {"EvaluationConfig", "EvaluationResult", "evaluate"}:
        from .evaluation import EvaluationConfig, EvaluationResult, evaluate

        return {
            "EvaluationConfig": EvaluationConfig,
            "EvaluationResult": EvaluationResult,
            "evaluate": evaluate,
        }[name]
    if name in {"VideoAnalysisConfig", "VideoAnalysisResult", "analyze_video"}:
        from .video import VideoAnalysisConfig, VideoAnalysisResult, analyze_video

        return {
            "VideoAnalysisConfig": VideoAnalysisConfig,
            "VideoAnalysisResult": VideoAnalysisResult,
            "analyze_video": analyze_video,
        }[name]
    raise AttributeError(name)
