from __future__ import annotations

import contextlib
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .backbones import YOLOBackboneEncoder
from .config import PretrainConfig
from .data import UnlabeledImageDataset, build_transform, discover_images
from .registry import build_method
from .utils import (
    DistributedContext,
    cleanup_distributed,
    cosine_momentum,
    initialize_distributed,
    reduce_mean,
    seed_everything,
)


@dataclass(frozen=True)
class PretrainResult:
    output_dir: Path
    yolo_checkpoint: Path
    ssl_checkpoint: Path
    history_csv: Path
    manifest_json: Path


def _move_batch(batch, device: torch.device):
    if isinstance(batch, (tuple, list)):
        return tuple(item.to(device, non_blocking=True) for item in batch)
    return batch.to(device, non_blocking=True)


def _make_scheduler(optimizer, config: PretrainConfig, optimizer_steps: int):
    warmup_steps = config.warmup_epochs * max(1, optimizer_steps // config.epochs)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-3, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, optimizer_steps - warmup_steps - 1)
        minimum = config.min_learning_rate / config.learning_rate
        return minimum + 0.5 * (1.0 - minimum) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _accumulation_group_size(batch_index: int, total_batches: int, steps: int) -> int:
    """Return the true divisor for this accumulation group, including a short final group."""

    remainder = total_batches % steps
    return remainder if remainder and batch_index > total_batches - remainder else steps


def _write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pretrain(config: PretrainConfig) -> PretrainResult:
    """Pretrain a YOLO backbone with one of the supported SSL objectives."""

    config.validate()
    distributed = initialize_distributed()
    try:
        return _pretrain(config, distributed)
    finally:
        # This also covers data/model setup failures that happen before the epoch loop.
        cleanup_distributed()


def _pretrain(config: PretrainConfig, distributed: DistributedContext) -> PretrainResult:
    seed_everything(config.seed, distributed.rank)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Imported lazily so the educational loss modules remain independently testable.
    from ultralytics import YOLO

    image_paths = discover_images(config.image_roots, config.max_images, config.seed)
    dataset = UnlabeledImageDataset(image_paths, build_transform(config))
    if config.method in {"simclr", "byol", "moco", "dinov2"} and len(dataset) < 2:
        raise ValueError(f"{config.method} requires at least two images")
    if config.method == "simclr" and math.ceil(len(dataset) / distributed.world_size) < 2:
        raise ValueError("SimCLR requires at least two images per distributed process")

    sampler = None
    if distributed.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
        )
    drop_last = len(dataset) >= config.batch_size * distributed.world_size
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.workers,
        pin_memory=distributed.device.type == "cuda",
        persistent_workers=config.workers > 0,
        drop_last=drop_last,
    )
    if not loader:
        raise RuntimeError("The SSL DataLoader has no batches")

    yolo = YOLO(config.yolo_model)
    encoder = YOLOBackboneEncoder(yolo.model).to(distributed.device)
    method = build_method(config, encoder, distributed.device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in method.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(len(loader) / config.grad_accum_steps)
    total_optimizer_steps = config.epochs * steps_per_epoch
    scheduler = _make_scheduler(optimizer, config, total_optimizer_steps)
    use_amp = bool(config.amp and distributed.device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    global_step = 0
    best_loss = float("inf")
    history: list[dict] = []
    if config.resume:
        state = torch.load(config.resume, map_location="cpu", weights_only=True)
        method.load_state_dict(state["method"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", 0))
        best_loss = float(state.get("best_loss", best_loss))
        history = list(state.get("history", []))

    train_model = method
    if distributed.world_size > 1:
        train_model = DistributedDataParallel(
            method,
            device_ids=[distributed.local_rank] if distributed.device.type == "cuda" else None,
            broadcast_buffers=True,
        )

    last_checkpoint = output_dir / "last_ssl.pt"
    best_checkpoint = output_dir / "best_ssl.pt"
    history_path = output_dir / "history.csv"
    source_stem = Path(config.yolo_model).stem
    yolo_path = output_dir / f"{config.method}_pretrained_{source_stem}.pt"
    manifest_path = output_dir / "run_manifest.json"

    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(start_epoch, config.epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            train_model.train()
            epoch_loss = 0.0
            batches = 0
            started = time.perf_counter()
            progress = tqdm(
                loader,
                disable=not distributed.is_main,
                desc=f"{config.method.upper()} {epoch:02d}/{config.epochs:02d}",
            )

            for batch_index, batch in enumerate(progress, start=1):
                batch = _move_batch(batch, distributed.device)
                accumulation_size = _accumulation_group_size(
                    batch_index,
                    len(loader),
                    config.grad_accum_steps,
                )
                is_update = (
                    batch_index % config.grad_accum_steps == 0 or batch_index == len(loader)
                )
                sync_context = contextlib.nullcontext()
                if isinstance(train_model, DistributedDataParallel) and not is_update:
                    sync_context = train_model.no_sync()

                momentum = cosine_momentum(
                    global_step,
                    total_optimizer_steps,
                    config.momentum,
                    config.final_momentum,
                )
                method.set_momentum(momentum)
                with sync_context:
                    with torch.amp.autocast(
                        device_type=distributed.device.type,
                        enabled=use_amp,
                    ):
                        raw_loss = train_model(batch)
                        scaled_loss = raw_loss / accumulation_size
                    if not torch.isfinite(raw_loss):
                        raise FloatingPointError(
                            f"Non-finite SSL loss: {float(raw_loss.detach())}"
                        )
                    scaler.scale(scaled_loss).backward()

                epoch_loss += float(raw_loss.detach())
                batches += 1
                if is_update:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(method.parameters(), config.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    method.after_optimizer_step()
                    scheduler.step()
                    global_step += 1

                if distributed.is_main:
                    progress.set_postfix(
                        loss=f"{epoch_loss / batches:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                        ema=f"{momentum:.5f}",
                    )

            mean_loss = reduce_mean(epoch_loss / max(1, batches), distributed.device)
            row = {
                "epoch": epoch,
                "loss": mean_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "ema_momentum": momentum,
                "seconds": time.perf_counter() - started,
            }
            history.append(row)

            if distributed.is_main:
                state = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_loss": min(best_loss, mean_loss),
                    "method": method.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "history": history,
                    "config": config.to_dict(),
                }
                if epoch % config.save_every == 0 or epoch == config.epochs:
                    torch.save(state, last_checkpoint)
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    torch.save(state, best_checkpoint)
                _write_history(history_path, history)
                print(
                    f"Epoch {epoch:02d} | loss={mean_loss:.5f} | "
                    f"time={row['seconds']:.1f}s"
                )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if distributed.is_main:
            yolo.save(str(yolo_path))
            if config.method == "dinov3":
                initialization = (
                    "YOLO student distilled without labels from a frozen pretrained "
                    f"{config.dinov3_model} teacher"
                )
            elif config.yolo_model.endswith(".pt"):
                initialization = (
                    "COCO-supervised warm start followed by label-free domain adaptation"
                )
            else:
                initialization = "random initialization; strict label-free SSL pretraining"
            manifest = {
                "method": config.method,
                "initialization": initialization,
                "source_model": config.yolo_model,
                "unlabeled_images": len(image_paths),
                "world_size": distributed.world_size,
                "per_gpu_batch_size": config.batch_size,
                "effective_batch_size": (
                    config.batch_size * distributed.world_size * config.grad_accum_steps
                ),
                "best_loss": best_loss,
                "config": config.to_dict(),
                "outputs": {
                    "yolo_checkpoint": str(yolo_path),
                    "ssl_checkpoint": str(last_checkpoint),
                    "history_csv": str(history_path),
                },
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    finally:
        cleanup_distributed()

    return PretrainResult(output_dir, yolo_path, last_checkpoint, history_path, manifest_path)
