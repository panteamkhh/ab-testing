"""Download and load the Cookie Cats experiment data."""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

BOOL_COLS = ["retention_1", "retention_7"]


def download(cfg: Config, force: bool = False) -> Path:
    dest = cfg.path("data", "raw_file")
    if dest.exists() and not force and dest.stat().st_size > 0:
        log.info("Raw data already present: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = cfg.get("data", "url")
    log.info("Downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310
    tmp.replace(dest)
    return dest


def load(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.path("data", "raw_file"))
    for col in BOOL_COLS:
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
    df["sum_gamerounds"] = pd.to_numeric(df["sum_gamerounds"], errors="coerce")
    df = df.drop_duplicates(subset=["userid"]).reset_index(drop=True)
    return df


def split(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    control = df[df["version"] == cfg.get("experiment", "control")].copy()
    treatment = df[df["version"] == cfg.get("experiment", "treatment")].copy()
    return control, treatment
