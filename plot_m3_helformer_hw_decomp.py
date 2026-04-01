from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import ensure_src_on_path

ensure_src_on_path()

from hybridts.data.m3 import M3_P  # noqa: E402
from hybridts.models.helformer import helformer_hw_components  # noqa: E402
from hybridts.viz import (  # noqa: E402
    save_helformer_hw_components_plot,
    save_helformer_hw_decomposition_pair_plot,
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def _default_out_dir() -> Path:
    legacy = Path("src/outputs/m3_eval/viz/06_helformer_hw_decomp")
    if legacy.parent.parent.exists():
        return legacy
    return Path("outputs/m3_eval/viz/06_helformer_hw_decomp")


def _norm_cat(cat: str) -> str:
    cat = str(cat).strip().lower()
    if cat in {"mounthly", "monthy", "mthly"}:
        return "monthly"
    return cat


def _parse_values(raw: object) -> np.ndarray:
    text = str(raw).strip().replace(";", ",")
    if not text:
        return np.zeros(0, dtype=float)
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except Exception:
            continue
    return np.asarray(out, dtype=float)


@dataclass(frozen=True)
class M3Series:
    category: str
    series_id: str
    y_tr: np.ndarray
    y_te: np.ndarray


def _load_m3_series(cat: str, series_id: str, *, csv_dir: Path) -> M3Series:
    cat = _norm_cat(cat)
    if cat not in M3_P:
        raise ValueError(f"Unknown M3 category '{cat}' (expected: {', '.join(sorted(M3_P))})")
    train_csv = Path(csv_dir) / f"M3_{cat}_TRAIN.csv"
    test_csv = Path(csv_dir) / f"M3_{cat}_TSTS.csv"
    if not (train_csv.exists() and test_csv.exists()):
        raise FileNotFoundError(f"Missing M3 CSVs for '{cat}': {train_csv} / {test_csv}")

    tr = pd.read_csv(train_csv, sep=None, engine="python")
    te = pd.read_csv(test_csv, sep=None, engine="python")
    tr.columns = [c.strip().lower() for c in tr.columns]
    te.columns = [c.strip().lower() for c in te.columns]
    if "series" not in tr.columns or "series" not in te.columns:
        raise ValueError(f"Unexpected M3 CSV schema for '{cat}' (missing 'series' column)")
    val_col_tr = "values" if "values" in tr.columns else tr.columns[-1]
    val_col_te = "values" if "values" in te.columns else te.columns[-1]

    series_id = str(series_id).strip()
    tr_row = tr.loc[tr["series"].astype(str) == series_id]
    te_row = te.loc[te["series"].astype(str) == series_id]
    if tr_row.empty or te_row.empty:
        raise ValueError(f"Series '{series_id}' not found in M3 '{cat}' CSVs")

    y_tr = _parse_values(tr_row.iloc[0][val_col_tr])
    y_te = _parse_values(te_row.iloc[0][val_col_te])
    if y_tr.size < 4 or y_te.size < 1:
        raise ValueError(f"Series '{series_id}' in '{cat}' is too short (train={y_tr.size}, test={y_te.size})")

    return M3Series(category=cat, series_id=series_id, y_tr=y_tr, y_te=y_te)


def _parse_series_spec(spec: str) -> tuple[str, str]:
    raw = str(spec).strip()
    if ":" in raw:
        cat, sid = raw.split(":", 1)
        return _norm_cat(cat), sid.strip()
    parts = raw.split()
    if len(parts) == 2:
        return _norm_cat(parts[0]), parts[1].strip()
    raise ValueError(f"Bad --series value '{spec}'. Use 'category:ID' (e.g. yearly:T170).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Helformer Holt-Winters (HW) decomposition components for selected M3 series",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/m3_eval.json"),
        help="Path to m3 eval JSON config (HW params).",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("src/data/m3/csv"),
        help="Directory containing normalized M3 CSVs.",
    )
    parser.add_argument(
        "--kind",
        choices=("pair", "components", "both"),
        default="both",
        help="Which plot(s) to generate: baseline+ratio pair, decomposition components, or both.",
    )
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        help="Series spec 'category:ID' (repeatable), e.g. yearly:T170",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNG files (defaults under src/outputs/m3_eval).",
    )
    args = parser.parse_args()

    cfg = _load_json(args.config)
    hw_trend = str(cfg.get("helformer_hw_trend", "mul"))
    hw_seasonal = str(cfg.get("helformer_hw_seasonal", "mul"))

    series_specs = args.series or ["yearly:T170", "quarterly:T181", "monthly:T378"]
    requested: list[tuple[str, str]] = [_parse_series_spec(s) for s in series_specs]

    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    for cat, sid in requested:
        item = _load_m3_series(cat, sid, csv_dir=args.csv_dir)
        season_period = M3_P[item.category]
        comps = helformer_hw_components(
            item.y_tr,
            item.y_te,
            seasonal_period=season_period,
            hw_trend=hw_trend,
            hw_seasonal=hw_seasonal,
        )
        if comps is None:
            raise SystemExit(f"HW baseline failed for M3 {item.category}:{item.series_id}")

        base_title = (
            f"M3 {item.category} {item.series_id} Helformer HW decomposition "
            f"(trend={comps.trend_mode}, seasonal={comps.seasonal_mode}, sp={season_period})"
        )
        prefix = f"{item.category}_{item.series_id}"

        if args.kind in {"pair", "both"}:
            save_path = out_dir / f"{prefix}.png"
            save_helformer_hw_decomposition_pair_plot(
                y_tr=item.y_tr,
                y_te=item.y_te,
                base_train=comps.base_train,
                base_test=comps.base_test,
                title=base_title,
                save_path=save_path,
            )
            print(f"[saved] {save_path}")

        if args.kind in {"components", "both"}:
            save_path = out_dir / f"{prefix}_components.png"
            save_helformer_hw_components_plot(
                y_tr=item.y_tr,
                y_te=item.y_te,
                base_train=comps.base_train,
                base_test=comps.base_test,
                season=comps.season,
                seasonal_mode=comps.seasonal_mode,
                title=base_title,
                save_path=save_path,
            )
            print(f"[saved] {save_path}")


if __name__ == "__main__":
    main()

