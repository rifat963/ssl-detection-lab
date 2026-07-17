"""Student-friendly helpers for reproducible SSL dry runs and DDP launching."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .config import PretrainConfig


@dataclass(frozen=True)
class DistributedPretrainResult:
    config_path: Path
    output_dir: Path
    command: tuple[str, ...]
    return_code: int
    seconds: float

    @property
    def succeeded(self) -> bool:
        return self.return_code == 0 and (self.output_dir / "run_manifest.json").exists()


def make_dry_run_config(
    method: str,
    train_images: str | Path,
    output_dir: str | Path,
    *,
    yolo_model: str = "yolo26n.yaml",
    max_images: int = 32,
    image_size: int = 128,
    batch_size: int = 4,
    workers: int = 2,
    seed: int = 42,
    dinov3_weights: str | None = None,
    dinov3_model: str = "dinov3_vits16",
) -> PretrainConfig:
    """Create a one-epoch, low-memory configuration for pipeline verification.

    This configuration is intentionally too small for scientific conclusions. Its
    purpose is to check data loading, augmentation, distributed training, checkpointing,
    and backbone transfer before launching a full experiment.
    """

    return PretrainConfig(
        method=method,
        image_roots=[str(train_images)],
        output_dir=str(output_dir),
        yolo_model=yolo_model,
        epochs=1,
        batch_size=batch_size,
        image_size=image_size,
        workers=workers,
        max_images=max_images,
        seed=seed,
        learning_rate=3e-4,
        min_learning_rate=3e-6,
        weight_decay=1e-4,
        warmup_epochs=0,
        grad_accum_steps=1,
        amp=True,
        projection_dim=128,
        hidden_dim=256,
        temperature=0.2,
        momentum=0.996,
        final_momentum=1.0,
        queue_size=512,
        dino_output_dim=1024,
        student_temperature=0.10,
        teacher_temperature=0.04,
        center_momentum=0.90,
        koleo_weight=0.10,
        local_crops=2,
        local_crop_size=min(64, image_size),
        dinov3_model=dinov3_model,
        dinov3_weights=dinov3_weights,
        mask_ratio=0.60,
        num_target_blocks=2,
        predictor_depth=1,
        predictor_heads=4,
        save_every=1,
    ).validate()


def launch_distributed_pretrain(
    config: PretrainConfig,
    *,
    num_processes: int = 2,
    config_path: str | Path | None = None,
    check: bool = True,
) -> DistributedPretrainResult:
    """Launch the normal pretraining CLI with one process per GPU."""

    config.validate()
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = (
        Path(config_path) if config_path is not None else output_dir / "pretrain_config.yaml"
    )
    config.save_yaml(destination)
    command = (
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={num_processes}",
        "-m",
        "ssldet.cli",
        "--config",
        str(destination),
    )
    started = time.perf_counter()
    completed = subprocess.run(command, check=False)
    result = DistributedPretrainResult(
        config_path=destination,
        output_dir=output_dir,
        command=command,
        return_code=completed.returncode,
        seconds=time.perf_counter() - started,
    )
    if check and not result.succeeded:
        raise RuntimeError(
            f"Distributed pretraining failed with return code {result.return_code}. "
            f"Inspect the output above and configuration at {destination}."
        )
    return result
