from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

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
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    crop = transforms.RandomResizedCrop(
        config.image_size,
        scale=(0.50, 1.0) if config.method in {"mae", "ijepa"} else (0.30, 1.0),
        ratio=(0.75, 1.3333),
        antialias=True,
    )

    if config.method in {"mae", "ijepa"}:
        # A single view: I-JEPA does not need two hand-designed positive views.
        return transforms.Compose([
            crop,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    strong_view = transforms.Compose([
        crop,
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.35, 0.35, 0.35, 0.06),
        ], p=0.8),
        transforms.RandomGrayscale(p=0.10),
        transforms.RandomApply([GaussianBlur()], p=0.50),
        transforms.ToTensor(),
        normalize,
    ])
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

