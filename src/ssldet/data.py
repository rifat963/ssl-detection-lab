from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .config import PretrainConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GaussianBlur:
    def __init__(self, sigma: tuple[float, float] = (0.1, 2.0)) -> None:
        self.sigma = sigma

    def __call__(self, image: Image.Image) -> Image.Image:
        radius = random.uniform(*self.sigma)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))


class TwoViews:
    def __init__(self, transform: Callable) -> None:
        self.transform = transform

    def __call__(self, image: Image.Image):
        return self.transform(image), self.transform(image)


class MultiCropViews:
    """Create two global crops and several lower-resolution local crops."""

    def __init__(self, global_transform: Callable, local_transform: Callable, local_crops: int):
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.local_crops = local_crops

    def __call__(self, image: Image.Image):
        global_views = (self.global_transform(image), self.global_transform(image))
        local_views = tuple(self.local_transform(image) for _ in range(self.local_crops))
        return global_views + local_views


def discover_images(roots: list[str], max_images: int | None, seed: int) -> list[Path]:
    paths: list[Path] = []
    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            raise FileNotFoundError(f"Image root does not exist: {root}")
        paths.extend(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No supported images found under: {roots}")
    if max_images is not None and len(paths) > max_images:
        paths = random.Random(seed).sample(paths, max_images)
        paths.sort()
    return paths


def build_transform(config: PretrainConfig):
    normalize = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    crop = v2.RandomResizedCrop(
        config.image_size,
        scale=(0.50, 1.0) if config.method in {"mae", "ijepa", "dinov3"} else (0.30, 1.0),
        ratio=(0.75, 1.3333),
        antialias=True,
    )

    if config.method in {"mae", "ijepa", "dinov3"}:
        # Single-view objectives do not need hand-designed positive pairs.
        return v2.Compose([
            crop,
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ])

    def strong_transform(resized_crop):
        return v2.Compose([
            resized_crop,
            v2.RandomHorizontalFlip(),
            v2.RandomApply([
                v2.ColorJitter(0.35, 0.35, 0.35, 0.06),
            ], p=0.8),
            v2.RandomGrayscale(p=0.10),
            v2.RandomApply([GaussianBlur()], p=0.50),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ])

    if config.method == "dinov2":
        global_crop = v2.RandomResizedCrop(
            config.image_size,
            scale=(0.32, 1.0),
            ratio=(0.75, 1.3333),
            antialias=True,
        )
        local_crop = v2.RandomResizedCrop(
            config.local_crop_size,
            scale=(0.05, 0.32),
            ratio=(0.75, 1.3333),
            antialias=True,
        )
        return MultiCropViews(
            strong_transform(global_crop),
            strong_transform(local_crop),
            config.local_crops,
        )

    strong_view = strong_transform(crop)
    return TwoViews(strong_view)


class UnlabeledImageDataset(Dataset):
    """Reads image pixels only. Annotation files are never opened."""

    def __init__(self, image_paths: list[Path], transform: Callable) -> None:
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image)
