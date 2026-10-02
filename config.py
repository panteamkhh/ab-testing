"""Configuration loading and path resolution."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def seed(self) -> int:
        return int(self.raw.get("project", {}).get("seed", 42))

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path(self, *keys: str) -> Path:
        rel = self.get(*keys)
        if rel is None:
            raise KeyError(f"Missing config key: {keys}")
        return (PROJECT_ROOT / rel).resolve()

    def ensure_dirs(self) -> None:
        for key in ("figures", "tables"):
            self.path("paths", key).mkdir(parents=True, exist_ok=True)
        self.path("data", "raw_file").parent.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(raw=data or {})
