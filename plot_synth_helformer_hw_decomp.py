from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from project_paths import ensure_src_on_path

ensure_src_on_path()

from hybridts.models.helformer import helformer_hw_components, helformer_hw_decompose  # noqa: E402
from hybridts.pipelines.synth_eval import PROFILES, _generate_series  # noqa: E402
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
    legacy = Path("src/outputs/synth_eval/viz/06_helformer_hw_decomp")
    if legacy.exists():
        return legacy
    return Path("outputs/synth_eval/viz/06_helformer_hw_decomp")


def _generate_series_map(
    *,
    profiles: list[str],
    n_per_profile: int,
    length: int,
    horizon: int,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, int | None]]:
    rng = np.random.default_rng(int(seed))
    out: dict[str, tuple[np.ndarray, np.ndarray, int | None]] = {}
    for profile_name in profiles:
        profile = PROFILES.get(profile_name)
        if profile is None:
            continue
        for idx in range(int(n_per_profile)):
            series_id = f"{profile.name}_{idx+1}"
            y = _generate_series(length=length, profile=profile, rng=rng)
            if y.size <= horizon + 8:
                continue
            out[series_id] = (y[:-horizon], y[-horizon:], profile.season_period)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Helformer Holt-Winters (HW) ratio decomposition for synthetic series",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synth_eval.json"),
        help="Path to synth eval JSON config (seed/length/horizon/profiles/n_per_profile, HW params).",
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
        help="Series id(s) to plot, e.g. trend_season_1 (repeat flag to pass multiple).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNG files (defaults near existing synth_eval outputs).",
    )
    args = parser.parse_args()

    cfg = _load_json(args.config)
    profiles = cfg.get("profiles") or []
    if not isinstance(profiles, list) or not profiles:
        profiles = list(PROFILES.keys())

    n_per_profile = int(cfg.get("n_per_profile", 1))
    length = int(cfg.get("length", 400))
    horizon = int(cfg.get("horizon", 24))
    seed = int(cfg.get("seed", 42))
    hw_trend = str(cfg.get("helformer_hw_trend", "mul"))
    hw_seasonal = str(cfg.get("helformer_hw_seasonal", "mul"))

    series_ids = args.series or ["trend_season_1", "trend_season_high_noise_1"]
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    series_map = _generate_series_map(
        profiles=profiles,
        n_per_profile=n_per_profile,
        length=length,
        horizon=horizon,
        seed=seed,
    )

    missing = [sid for sid in series_ids if sid not in series_map]
    if missing:
        available = ", ".join(sorted(series_map.keys()))
        raise SystemExit(
            f"Unknown series id(s): {', '.join(missing)}.\nAvailable: {available}\n"
        )

    for sid in series_ids:
        y_tr, y_te, season_period = series_map[sid]
        title = f"{sid} Helformer HW decomposition (trend={hw_trend}, seasonal={hw_seasonal}, sp={season_period})"

        if args.kind in {"pair", "both"}:
            decomp = helformer_hw_decompose(
                y_tr,
                y_te,
                seasonal_period=season_period,
                hw_trend=hw_trend,
                hw_seasonal=hw_seasonal,
            )
            if decomp is None:
                raise SystemExit(
                    f"HW baseline failed for {sid} (statsmodels missing or convergence issue)."
                )
            base_train, base_test, _ratio_train, _ratio_test = decomp
            save_path = out_dir / f"{sid}.png"
            save_helformer_hw_decomposition_pair_plot(
                y_tr=y_tr,
                y_te=y_te,
                base_train=base_train,
                base_test=base_test,
                title=title,
                save_path=save_path,
            )
            print(f"[saved] {save_path}")

        if args.kind in {"components", "both"}:
            comps = helformer_hw_components(
                y_tr,
                y_te,
                seasonal_period=season_period,
                hw_trend=hw_trend,
                hw_seasonal=hw_seasonal,
            )
            if comps is None:
                raise SystemExit(
                    f"HW baseline failed for {sid} (statsmodels missing or convergence issue)."
                )
            save_path = out_dir / f"{sid}_components.png"
            save_helformer_hw_components_plot(
                y_tr=y_tr,
                y_te=y_te,
                base_train=comps.base_train,
                base_test=comps.base_test,
                season=comps.season,
                seasonal_mode=comps.seasonal_mode,
                title=title,
                save_path=save_path,
            )
            print(f"[saved] {save_path}")


if __name__ == "__main__":
    main()
