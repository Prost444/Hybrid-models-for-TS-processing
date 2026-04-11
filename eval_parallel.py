"""CLI entry point for parallel benchmark evaluation."""
from __future__ import annotations

import argparse
import json
import warnings
from inspect import signature
from pathlib import Path

warnings.filterwarnings("ignore", message=".*np.object.*", category=FutureWarning)

from project_paths import ensure_src_on_path
ensure_src_on_path()

from hybridts.pipelines.parallel_benchmark import run_parallel_benchmark

DEFAULT_CONFIG = Path("configs/parallel_m3_full.json")


def load_config(path: Path | None):
    if path is None or not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def main():
    parser = argparse.ArgumentParser(description="Parallel benchmark evaluation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    sig = signature(run_parallel_benchmark)
    allowed = {k for k in sig.parameters}
    filtered = {k: v for k, v in cfg.items() if k in allowed}
    run_parallel_benchmark(**filtered)


if __name__ == "__main__":
    main()
