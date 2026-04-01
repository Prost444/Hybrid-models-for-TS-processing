"""CLI entry point for M3 benchmark evaluation."""
from __future__ import annotations

import argparse
import json
import warnings
from inspect import signature
from pathlib import Path

warnings.filterwarnings("ignore", message=".*np.object.*", category=FutureWarning)

from project_paths import ensure_src_on_path

ensure_src_on_path()

from hybridts.pipelines.m3_benchmark import run_m3_benchmark

DEFAULT_CONFIG = Path("configs/m3_smoke.json")


def load_config(path: Path | None):
    if path is None or not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def main():
    parser = argparse.ArgumentParser(description="M3 benchmark evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to a JSON config file (default: configs/m3_smoke.json)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    sig = signature(run_m3_benchmark)
    allowed = {k for k in sig.parameters}
    filtered = {k: v for k, v in cfg.items() if k in allowed}

    run_m3_benchmark(**filtered)


if __name__ == "__main__":
    main()
