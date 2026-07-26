"""Unified command line interface for catalog, evaluation, and video analysis."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any

from .catalog import capabilities


def _device(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def _video_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def _catalog_text() -> str:
    catalog = capabilities()
    lines = ["Supported SSL architectures", ""]
    for item in catalog["ssl_architectures"]:
        lines.append(
            f"- {item['name'].upper():7} | {item['category']:15} | {item['objective']}"
        )
    lines.extend(["", "Supported model families", ""])
    for item in catalog["model_families"]:
        ssl = "SSL + evaluation + video" if item["ssl_backbone"] else "evaluation + video"
        scales = ",".join(item["scales"]) or "custom"
        lines.append(f"- {item['name']:25} | scales: {scales:11} | {ssl}")
        if item["notes"]:
            lines.append(f"  {item['notes']}")
    lines.extend(["", "Official DINOv2 reference feature backbones", ""])
    for item in catalog["dinov2_feature_backbones"]:
        register_note = "with registers" if item["registers"] else "without registers"
        lines.append(
            f"- {item['name']:25} | {item['architecture']:9} | "
            f"dim {item['embedding_dim']:4} | {register_note}"
        )
    lines.extend(["", "Official DINOv3 feature backbones", ""])
    for item in catalog["dinov3_feature_backbones"]:
        lines.append(
            f"- {item['name']:25} | {item['architecture']:15} | "
            f"dim {item['embedding_dim']:4} | user-supplied weights"
        )
    lines.extend(["", "Object-detection backends", ""])
    for item in catalog["object_detection_backends"]:
        commercial = "enterprise option available" if item["commercial_license_available"] else ""
        lines.append(
            f"- {item['name']:25} | {item['open_source_license']:10} | {commercial}"
        )
    lines.extend(
        [
            "",
            "DINOv2/DINOv3 feature backbones are not standalone detectors. Attach a",
            "detection head or use a fine-tuned detector before video analysis.",
        ]
    )
    return "\n".join(lines)


def _add_common_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Model family/name, e.g. yolo26n")
    parser.add_argument(
        "--weights", required=True, help="Local/custom .pt weights or official name"
    )
    parser.add_argument("--output", required=True, help="Output report directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--device", type=_device, default=None, help="cpu, mps, 0, 0,1, etc.")
    parser.add_argument(
        "--max-det", type=int, default=300, help="Maximum detections per frame/image"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ssldet",
        description="Inspect SSL support, evaluate labelled data, or analyse a video.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    models = subparsers.add_parser("models", help="Show supported models and SSL architectures")
    models.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    subparsers.add_parser("doctor", help="Show dependency, CUDA, and GPU compatibility")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate weights on a labelled dataset")
    _add_common_model_arguments(evaluate)
    evaluate.add_argument("--data", required=True, help="Ultralytics dataset YAML")
    evaluate.add_argument("--split", choices=("train", "val", "test"), default="val")
    evaluate.add_argument("--batch", type=int, default=16)
    evaluate.add_argument("--workers", type=int, default=8)
    evaluate.add_argument("--half", action="store_true")
    evaluate.add_argument("--no-plots", action="store_true")

    video = subparsers.add_parser("video", help="Analyse a local video, URL, or stream")
    _add_common_model_arguments(video)
    video.add_argument(
        "--source",
        required=True,
        type=_video_source,
        help="Video path, webcam index, HTTP/RTSP link, or YouTube URL",
    )
    video.add_argument(
        "--tracker", default="botsort.yaml", help="Tracker YAML; use 'none' to disable"
    )
    video.add_argument("--stride", type=int, default=1, help="Analyse every Nth frame")
    video.add_argument("--max-frames", type=int, default=None)
    video.add_argument("--no-save-video", action="store_true")
    video.add_argument("--save-txt", action="store_true")
    video.add_argument("--save-conf", action="store_true")
    return parser


def models_main() -> None:
    """Dedicated entry point for installations that only want the support list."""

    parser = argparse.ArgumentParser(prog="ssldet-models")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    print(json.dumps(capabilities(), indent=2) if args.json else _catalog_text())


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_paths(item) for item in value]
    return str(value)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "models":
        print(json.dumps(capabilities(), indent=2) if args.json else _catalog_text())
        return
    if args.command == "doctor":
        from .runtime import runtime_report

        print(json.dumps(runtime_report(), indent=2))
        return

    if args.command == "evaluate":
        from .evaluation import EvaluationConfig, evaluate

        result = evaluate(
            EvaluationConfig(
                model_name=args.model,
                weights_file=args.weights,
                data=args.data,
                output_dir=args.output,
                split=args.split,
                image_size=args.imgsz,
                batch_size=args.batch,
                confidence=0.001 if args.conf is None else args.conf,
                iou=args.iou,
                max_detections=args.max_det,
                device=args.device,
                workers=args.workers,
                half=args.half,
                plots=not args.no_plots,
            )
        )
    else:
        from .video import VideoAnalysisConfig, analyze_video

        tracker = None if args.tracker.lower() == "none" else args.tracker
        result = analyze_video(
            VideoAnalysisConfig(
                video_source=args.source,
                model_name=args.model,
                weights_file=args.weights,
                output_dir=args.output,
                confidence=0.25 if args.conf is None else args.conf,
                iou=args.iou,
                image_size=args.imgsz,
                max_detections=args.max_det,
                device=args.device,
                tracker=tracker,
                video_stride=args.stride,
                max_frames=args.max_frames,
                save_annotated=not args.no_save_video,
                save_txt=args.save_txt,
                save_confidence=args.save_conf,
            )
        )
    print(json.dumps(_stringify_paths(asdict(result)), indent=2))


if __name__ == "__main__":
    main()
