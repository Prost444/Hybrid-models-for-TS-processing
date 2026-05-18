#!/usr/bin/env python3
"""Clean 'direct vs iterative' degradation bar chart (M3, approved thesis data).

Sorted by degradation so the conclusion is immediately visible:
classical ≈ no change → models with learnable decomposition degrade most.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "thesis" / "figures"

NAVY = "#1F2A4E"
ACCENT = "#E07A5F"

# Approved M3 rollout numbers from tab:rollout-m3 (chapter2.tex)
data = [
    ("Seasonal Naive", 14.94, 14.94),
    ("Auto-ARIMA",     15.79, 15.79),
    ("ETS",            13.89, 13.61),
    ("DLinear",        29.76, 30.16),
    ("PatchTST",       21.33, 23.16),
    ("N-BEATS",        18.55, 22.42),
    ("TimesNet",       17.47, 21.75),
    ("Autoformer",     23.08, 28.03),
    ("FEDformer",      22.34, 28.62),
]
# sort by degradation (rollout - direct) ascending
data.sort(key=lambda r: r[2] - r[1])
names = [d[0] for d in data]
direct = np.array([d[1] for d in data])
roll = np.array([d[2] for d in data])

plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(12.6, 5.6))
x = np.arange(len(names))
w = 0.38
b1 = ax.bar(x - w / 2, direct, w, label="Прямой прогноз", color=NAVY)
b2 = ax.bar(x + w / 2, roll, w, label="Итеративный прогноз", color=ACCENT)

for i, (d, r) in enumerate(zip(direct, roll)):
    delta = r - d
    if abs(delta) >= 0.5:
        ax.annotate(f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}",
                    (i + w / 2, r), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=10,
                    fontweight="bold", color="#C2503A")

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel("sMAPE (%)", fontsize=12)
ax.set_title("Деградация качества при переходе к итеративному прогнозу "
             "(M3, среднее по категориям)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="upper left")
ax.set_ylim(0, max(roll) * 1.18)
ax.grid(axis="y", alpha=0.25)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

plt.tight_layout()
fig.savefig(OUT / "rollout_degradation_bars.png", dpi=170,
            bbox_inches="tight", facecolor="white")
print("saved", OUT / "rollout_degradation_bars.png")
