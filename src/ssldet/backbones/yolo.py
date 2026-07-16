from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOBackboneEncoder(nn.Module):
    """Expose the YAML-defined Ultralytics backbone as a reusable encoder.

    The contained modules are references to the detector's real backbone.
    Updating this encoder therefore updates the detector that will later be saved.
    """

    def __init__(self, yolo_task_model: nn.Module) -> None:
        super().__init__()
        backbone_definition = getattr(yolo_task_model, "yaml", {}).get("backbone", [])
        if not backbone_definition:
            raise ValueError("The Ultralytics model has no YAML backbone definition")

        self.backbone_end = len(backbone_definition)
        self.layers = nn.ModuleList(list(yolo_task_model.model[: self.backbone_end]))
        self.save = set(getattr(yolo_task_model, "save", []))

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        saved_outputs: list[torch.Tensor | None] = []
        for module in self.layers:
            source = getattr(module, "f", -1)
            if source != -1:
                if isinstance(source, int):
                    x = saved_outputs[source]
                else:
                    x = [x if item == -1 else saved_outputs[item] for item in source]
            x = module(x)
            saved_outputs.append(x if getattr(module, "i", -1) in self.save else None)
        if isinstance(x, (list, tuple)):
            raise TypeError("Expected the final YOLO backbone output to be one feature tensor")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_feature_map(x)
        if features.ndim == 4:
            return F.adaptive_avg_pool2d(features, 1).flatten(1)
        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim == 2:
            return features
        raise ValueError(f"Unsupported feature shape: {tuple(features.shape)}")

    @torch.no_grad()
    def infer_dimensions(self, image_size: int, device: torch.device) -> tuple[int, int, int]:
        was_training = self.training
        self.eval()
        sample = torch.zeros(1, 3, image_size, image_size, device=device)
        feature_map = self.forward_feature_map(sample)
        self.train(was_training)
        if feature_map.ndim != 4:
            raise ValueError("MAE and I-JEPA require a spatial BxCxHxW feature map")
        return int(feature_map.shape[1]), int(feature_map.shape[2]), int(feature_map.shape[3])
