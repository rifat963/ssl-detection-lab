"""Runtime diagnostics for notebooks, command-line users, and bug reports."""

from __future__ import annotations

import platform
import re
from importlib.metadata import PackageNotFoundError, version
from typing import Any

MINIMUM_RUNTIME_VERSIONS = {
    "torch": "2.4",
    "torchvision": "0.19",
    "ultralytics": "8.4.96",
}


def _installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _numeric_version(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in re.findall(r"\d+", value)[:3]]
    return tuple((parts + [0, 0, 0])[:3])


def runtime_report() -> dict[str, Any]:
    """Return package, CUDA, and GPU information with compatibility checks."""

    packages = {
        name: {
            "installed": _installed_version(name),
            "minimum": minimum,
        }
        for name, minimum in MINIMUM_RUNTIME_VERSIONS.items()
    }
    for values in packages.values():
        installed = values["installed"]
        values["supported"] = bool(
            installed is not None
            and _numeric_version(installed) >= _numeric_version(values["minimum"])
        )

    cuda: dict[str, Any] = {
        "available": False,
        "runtime": None,
        "cudnn": None,
        "device_count": 0,
        "devices": [],
        "error": None,
    }
    try:
        import torch

        cuda["available"] = torch.cuda.is_available()
        cuda["runtime"] = torch.version.cuda
        cuda["cudnn"] = torch.backends.cudnn.version()
        cuda["device_count"] = torch.cuda.device_count()
        cuda["devices"] = [
            {
                "index": index,
                "name": (properties := torch.cuda.get_device_properties(index)).name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "memory_gib": round(properties.total_memory / 2**30, 2),
            }
            for index in range(torch.cuda.device_count())
        ]
    except (ImportError, OSError, RuntimeError) as error:
        # A broken binary/DLL or CUDA driver should be reportable through `doctor`, not
        # make the diagnostic command itself crash.
        cuda["error"] = f"{type(error).__name__}: {error}"
        packages["torch"]["supported"] = False

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "features": {
            "dinov3": {
                "minimum_torch": "2.7.1",
                "supported": bool(
                    packages["torch"]["installed"]
                    and _numeric_version(packages["torch"]["installed"]) >= (2, 7, 1)
                ),
                "weights": "user-supplied local path or authorized URL",
            }
        },
        "supported": all(values["supported"] for values in packages.values()),
    }


def assert_supported_runtime(
    *, require_cuda: bool = False, minimum_gpus: int = 0
) -> dict[str, Any]:
    """Validate dependency floors and optional CUDA requirements, then return the report."""

    report = runtime_report()
    problems = [
        f"{name} {values['installed'] or 'not installed'} (requires {values['minimum']}+)"
        for name, values in report["packages"].items()
        if not values["supported"]
    ]
    if require_cuda and not report["cuda"]["available"]:
        problems.append("CUDA is unavailable")
    if report["cuda"]["device_count"] < minimum_gpus:
        problems.append(
            f"found {report['cuda']['device_count']} GPU(s), requires at least {minimum_gpus}"
        )
    if problems:
        raise RuntimeError("Unsupported runtime: " + "; ".join(problems))
    return report
