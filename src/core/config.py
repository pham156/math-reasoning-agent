from __future__ import annotations

import os
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path("/home/pham156/math-reasoning-agent")
DEFAULT_SCRATCH_ROOT = Path("/scratch/gilbreth/pham156/math-reasoning-agent")


def project_root() -> Path:
    root = Path(os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT)).expanduser().resolve()
    return root


def scratch_root() -> Path:
    root = Path(os.environ.get("SCRATCH_ROOT", DEFAULT_SCRATCH_ROOT)).expanduser().resolve()
    ensure_scratch_path(root, "SCRATCH_ROOT")
    return root


def ensure_scratch_path(path: str | Path, name: str = "path") -> Path:
    resolved = Path(path).expanduser().resolve()
    scratch_prefix = Path("/scratch/gilbreth").resolve()
    if scratch_prefix not in (resolved, *resolved.parents):
        raise ValueError(f"{name} must be under /scratch/gilbreth, got: {resolved}")
    return resolved


def model_dir(name: str) -> Path:
    return scratch_root() / "models" / name


def data_dir(name: str = "") -> Path:
    base = scratch_root() / "data"
    return base / name if name else base


def output_dir(name: str = "") -> Path:
    base = scratch_root() / "outputs"
    return base / name if name else base


def cache_dir() -> Path:
    return scratch_root() / "cache"


def log_dir(name: str = "") -> Path:
    base = project_root() / "logs"
    return base / name if name else base


def print_storage_summary() -> None:
    root = project_root()
    scratch = scratch_root()
    locations = {
        "PROJECT_ROOT": root,
        "SCRATCH_ROOT": scratch,
        "models": scratch / "models",
        "data": scratch / "data",
        "outputs": scratch / "outputs",
        "cache": scratch / "cache",
        "logs": root / "logs",
    }
    for label, path in locations.items():
        if label in {"models", "data", "outputs", "cache", "SCRATCH_ROOT"}:
            ensure_scratch_path(path, label)
        print(f"{label}: {path}")
    print("Confirmed: large files are configured under /scratch/gilbreth; code/logs are in home.")
