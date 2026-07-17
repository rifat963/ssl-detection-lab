"""SSL pretraining, detector evaluation, and video analysis building blocks.

Heavy training/inference modules are imported lazily so the model catalog can be
displayed even before PyTorch and Ultralytics are installed.
"""

from .catalog import (
    DINOV2_FEATURE_BACKBONES,
    MODEL_FAMILIES,
    SSL_ARCHITECTURES,
    capabilities,
)

__all__ = [
    "DINOV2_FEATURE_BACKBONES",
    "MODEL_FAMILIES",
    "SSL_ARCHITECTURES",
    "EvaluationConfig",
    "EvaluationResult",
    "DistributedPretrainResult",
    "PretrainConfig",
    "PretrainResult",
    "VideoAnalysisConfig",
    "VideoAnalysisResult",
    "analyze_video",
    "build_dinov2_transform",
    "capabilities",
    "evaluate",
    "launch_distributed_pretrain",
    "load_dinov2_backbone",
    "make_dry_run_config",
    "pretrain",
]
__version__ = "0.3.1"


def __getattr__(name: str):
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
