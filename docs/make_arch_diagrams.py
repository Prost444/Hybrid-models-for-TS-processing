#!/usr/bin/env python3
"""Clean architecture pipeline diagrams for the neural-models slide.

Horizontal box-and-arrow pipelines (the style the supervisor approved),
navy + terracotta palette, Cyrillic via DejaVu Sans.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "thesis" / "figures"

NAVY = "#1F2A4E"
NAVY2 = "#34467A"
ACCENT = "#E07A5F"
LIGHT = "#EEF1F7"
INK = "#2B2B2B"

plt.rcParams["font.family"] = "DejaVu Sans"


X0, GAP, BW, BH = 0.0, 0.42, 2.18, 0.62
N = 5
XMAX = X0 + (N - 1) * (BW + GAP) + BW   # rightmost box edge


def pipeline(ax, y, title, boxes, accent_idx=None):
    ax.text(0.0, y + 0.52, title, fontsize=13, fontweight="bold",
            color=NAVY, va="center")
    n = len(boxes)
    for i, label in enumerate(boxes):
        x = X0 + i * (BW + GAP)
        is_acc = (accent_idx is not None and i in accent_idx)
        fc = ACCENT if is_acc else NAVY
        box = FancyBboxPatch((x, y - BH / 2), BW, BH,
                             boxstyle="round,pad=0.02,rounding_size=0.06",
                             linewidth=0, facecolor=fc)
        ax.add_patch(box)
        ax.text(x + BW / 2, y, label, ha="center", va="center",
                fontsize=10.5, color="white", fontweight="bold")
        if i < n - 1:
            ar = FancyArrowPatch((x + BW, y), (x + BW + GAP, y),
                                 arrowstyle="-|>", mutation_scale=15,
                                 linewidth=2.2, color=NAVY2)
            ax.add_patch(ar)


fig, ax = plt.subplots(figsize=(13.4, 6.7))
ax.set_xlim(-0.35, XMAX + 0.35)
ax.set_ylim(-0.1, 7.4)
ax.axis("off")

pipeline(ax, 6.5, "DLinear — линейная модель с декомпозицией",
         ["Вход\n(окно L)", "Скользящее\nсреднее",
          "Тренд + сезон\n(2 ветви)", "Линейные\nпроекции",
          "Прогноз H"], accent_idx=[1])

pipeline(ax, 4.75, "N-BEATS — обучаемое базисное разложение",
         ["Вход\n(окно L)", "Стек блоков\n(FC + ReLU)",
          "Базис: тренд /\nсезон / общий", "Двойной\nостаток",
          "Σ прогнозов"], accent_idx=[2])

pipeline(ax, 3.0, "TimesNet — 2D-моделирование периодов",
         ["Вход\n(окно L)", "БПФ:\nпериоды", "1D → 2D\nпо периодам",
          "2D-свёртки\n(Inception)", "Агрегация →\nпрогноз"],
         accent_idx=[2])

pipeline(ax, 1.25, "Helformer — гибрид: Холт–Уинтерс + нейросеть",
         ["Вход\n(окно L)", "Декомпозиция\nХолта–Уинтерса",
          "Отношения\nфакт / база", "Трансформер\n+ LSTM",
          "Коррекция →\nпрогноз"], accent_idx=[1, 3])

plt.tight_layout()
fig.savefig(OUT / "arch_neural_pipelines.png", dpi=170,
            bbox_inches="tight", facecolor="white")
print("saved", OUT / "arch_neural_pipelines.png")
