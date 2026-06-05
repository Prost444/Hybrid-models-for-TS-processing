#!/usr/bin/env python3
"""Render formula images (mathtext) for architecture/metric slides.

Tight transparent PNGs, navy text — embedded into the .pptx so formulas look
like Lykov's rendered-LaTeX cards (pptx itself cannot typeset math).
Latin subscripts only (mathtext does not render Cyrillic reliably).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / "assets" / "formulas"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#1B2649"

FORMULAS = {
    # ---- classical ----
    "arima": r"$w_t=\Delta^{d}y_t,\quad "
             r"w_t=c+\sum_{i=1}^{p}\varphi_i\,w_{t-i}"
             r"+\varepsilon_t+\sum_{j=1}^{q}\theta_j\,\varepsilon_{t-j}$",
    "ets":   r"$\hat{y}_{t+h}=l_t+h\,b_t+s_{t+h}$",
    "prophet": r"$y(t)=g(t)+s(t)+h(t)+\varepsilon_t$",
    # ---- metrics ----
    "smape": r"$\mathrm{sMAPE}=\dfrac{100\%}{N}\sum_{i=1}^{N}"
             r"\dfrac{|y_i-\hat{y}_i|}{|y_i|+|\hat{y}_i|}$",
    "mase":  r"$\mathrm{MASE}=\dfrac{\frac{1}{N}\sum_i|y_i-\hat{y}_i|}"
             r"{\frac{1}{T-m}\sum_t|y_t-y_{t-m}|}$",
    "rmse":  r"$\mathrm{RMSE}=\sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}$",
    # ---- neural ----
    "dlinear": r"$X=X_{trend}+X_{seasonal},\quad "
               r"\hat{Y}=W_{t}\,X_{trend}+W_{s}\,X_{seasonal}$",
    "nbeats":  r"$x_{\ell}=x_{\ell-1}-\hat{x}_{\ell-1},\qquad "
               r"\hat{y}=\sum_{\ell}\hat{y}_{\ell}$",
    "timesnet": r"$X^{1D}\to\mathrm{FFT}\to p_k=\dfrac{T}{k},\quad "
                r"X^{2D}_{k}=\mathrm{Reshape}_{p_k,f_k}(X^{1D})$",
    "autoformer": r"$\mathcal{R}_{xx}(\tau)=\dfrac{1}{L}\sum_{t}x_t\,x_{t-\tau},"
                  r"\quad X=\mathcal{T}+\mathcal{S}\ \ (\mathrm{SeriesDecomp})$",
    "helformer": r"$y_t=\mathrm{HW}(y)\;+\;\mathrm{Attention}(Q,K,V)$",
    # ---- economics ----
    "payback": r"$S=85\,000\cdot\dfrac{N(N+1)}{2}\;\geq\;890\,142{,}8$",
    "npv":     r"$\mathrm{NPV}=\sum_{i=1}^{n}\dfrac{CF_i}{(1+r)^{i}}-IC$",
}


def render(name, tex, fontsize=22):
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0.5, 0.5, tex, ha="center", va="center",
             fontsize=fontsize, color=NAVY)
    out = OUT / f"{name}.png"
    fig.savefig(out, dpi=220, transparent=True, bbox_inches="tight",
                pad_inches=0.06)
    plt.close(fig)
    print("saved", out.name)


if __name__ == "__main__":
    for k, v in FORMULAS.items():
        render(k, v)
