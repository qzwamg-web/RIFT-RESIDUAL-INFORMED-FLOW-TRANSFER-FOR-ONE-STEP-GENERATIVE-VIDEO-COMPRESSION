"""Config-driven paths for RAQVC.

All runtime paths come from a YAML config or explicit arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml

PathLike = Union[str, Path]


def repo_root() -> Path:
    """Return the absolute path to the RAQVC repo root."""
    # This file lives at <repo>/raqvc/utils/paths.py
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RAQVCPaths:
    """Runtime paths for RAQVC. All fields become absolute pathlib.Path objects."""

    artifacts_root: Path
    checkpoints_root: Path
    data_root: Path

    def __post_init__(self) -> None:
        # dataclass(frozen=True) blocks assignment; use object.__setattr__.
        for field in ("artifacts_root", "checkpoints_root", "data_root"):
            raw = getattr(self, field)
            p = Path(raw).expanduser()
            object.__setattr__(self, field, p.resolve() if p.exists() else p)


_REQUIRED_KEYS = ("artifacts_root", "checkpoints_root", "data_root")


def load_paths(config_path: PathLike) -> RAQVCPaths:
    """Load an RAQVCPaths from a YAML file.

    Required keys: artifacts_root, checkpoints_root, data_root.
    Missing keys raise KeyError. `~` in paths is expanded to the current user's home.
    """
    with Path(config_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for key in _REQUIRED_KEYS:
        if key not in data:
            raise KeyError(f"Required key missing from {config_path}: {key}")
    return RAQVCPaths(
        artifacts_root=data["artifacts_root"],
        checkpoints_root=data["checkpoints_root"],
        data_root=data["data_root"],
    )
