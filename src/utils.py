"""Pomocne funkcije: seed-ovi i evidencija okruzenja (reproduktivnost)."""
import json
import os
import platform
import random
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

TRACKED_PACKAGES = ["torch", "torchvision", "numpy", "pandas",
                    "scikit-learn", "mlflow", "optuna", "matplotlib"]


def set_seed(seed: int = 42) -> None:
    """Fiksira sve izvore nasumicnosti."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def get_env_info() -> dict:
    """Verzije biblioteka i podaci o masini na kojoj se trenira."""
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "git_commit": _git_commit(),
        "packages": versions,
    }


def save_env_info(path: str | Path) -> dict:
    info = get_env_info()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2))
    return info