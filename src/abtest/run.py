"""Entry point:  python run.py [--force-download]"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from abtest.config import load_config  # noqa: E402
from abtest.pipeline import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B testing pipeline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run(load_config(args.config), force_download=args.force_download)


if __name__ == "__main__":
    main()
