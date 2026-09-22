"""Three schematic/data figures for manuscript v11.

Palette matches the LaTeX build:
    rose  #DF81A5   (expensive direction / gate)
    blue  #4E7FA8   (address / reference)
    dark  #2C6A9B   (dual / optimal)

  figV.pdf     the erase V: locality vs residual, two branches, measured points
  figRank.pdf  rank collapse: read error vs load m, additive vs dual
  figDual.pdf  dual-vector geometry: a_j . e* = 1, a_p . e* = 0

Run: python experiments/make_fig_v11.py --outdir ../revision/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROSE = "#DF81A5"
BLUE = "#4E7FA8"
DARK = "#2C6A9B"
GREY = "#555555"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.edgecolor": GREY,
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": GREY,
    "ytick.color": GREY,
    "figure.dpi": 300,
})


def fig_v(outdir):
    """The V of Theorem 2 at dk = 128, slope s = L1 = 0.2575."""
    s = 0.2575
    r = np.linspace(0.0, 1.0, 200)
    left = s * (1.0 - r)          # c <= 1
    right = s * (1.0 + r)         # c > 1 (fold)

    # measured points from Table tab:v (dk = 128)
    meas = [(0.0000, 0.2575), (0.5000, 0.1288), (0.7500, 0.0644),
            (0.2500, 0.3219), (0.5000, 0.3863)]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(r, left, "-", color=BLUE, lw=1.4,
            label=r"$(1-r)/g_0,\ c\leq 1$")
    ax.plot(r, right, "--", color=ROSE, lw=1.4,
            label=r"$(1+r)/g_0$, past the fold")
    mx = [p[0] for p in meas]
    my = [p[1] for p in meas]
    ax.plot(mx, my, "o", ms=4.2, mfc="white", mec=DARK, mew=1.0,
            label="measured", zorder=5)
    ax.plot([0], [s], "o", ms=5.0, mfc=DARK, mec=DARK, zorder=6)
    ax.annotate("perfect erase\n$(\\varepsilon=0,L_1)$", xy=(0, s),
                xytext=(0.16, 0.36), fontsize=7.2, color=DARK,
                arrowprops=dict(arrowstyle="->", color=DARK, lw=0.7))
    ax.set_xlabel("residual  $\\varepsilon$")
    ax.set_ylabel("locality")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.01, 0.56)
    ax.legend(loc="upper center", frameon=False, fontsize=6.8,
              handlelength=1.6, borderaxespad=0.2)
    ax.set_title(r"Erase V at $d_k=128$, slope $s=L_1$", fontsize=8.5,
                 color="#222222", pad=4)
    fig.tight_layout(pad=0.3)
    fig.savefig(outdir / "figV.pdf")
    plt.close(fig)


def fig_rank(outdir):
    """Read error vs load m, dk = 64, additive vs dual (Table tab:cliff)."""
    m = np.array([32, 64, 128, 256])
    add = np.array([0.5986, 0.8775, 1.2346, 1.7616])
    dual = np.array([0.0, 0.0, 6.1481, 26.9102])

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.axvline(64, color=GREY, lw=0.7, ls=":")
    ax.semilogy(m, np.maximum(add, 5e-3), "-s", color=BLUE, ms=4,
                mfc="white", mew=1.0, lw=1.3, label="additive")
    ax.semilogy(m, np.maximum(dual, 5e-3), "-o", color=ROSE, ms=4,
                mfc="white", mew=1.0, lw=1.3, label="dual ($B=(AA^\\top)^{-1}A$)")
    ax.text(66, 4e-2, "$m=d_k$", fontsize=7, color=GREY)
    ax.annotate("rank collapse:\nundefined, not decay", xy=(128, 6.15),
                xytext=(120, 1.2e-2), fontsize=7, color=ROSE,
                ha="left", arrowprops=dict(arrowstyle="->", color=ROSE, lw=0.7))
    ax.set_xlabel("load  $m$")
    ax.set_ylabel("read error")
    ax.set_ylim(3e-3, 60)
    ax.set_xlim(10, 290)
    ax.legend(loc="lower right", frameon=False, fontsize=6.8,
              handlelength=1.6, borderaxespad=0.3)
    ax.set_title(r"Rank cliff at $d_k=64$", fontsize=8.5, color="#222222", pad=4)
    fig.tight_layout(pad=0.3)
    fig.savefig(outdir / "figRank.pdf")
    plt.close(fig)


def fig_dual(outdir):
    """Dual-vector geometry for two unit addresses a_j, a_p in R^2."""
    th = np.deg2rad(58.0)
    aj = np.array([1.0, 0.0])
    ap = np.array([np.cos(th), np.sin(th)])
    A = np.vstack([aj, ap])
    e = A.T @ np.linalg.solve(A @ A.T, np.array([1.0, 0.0]))   # dual of a_j

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    # unit circle
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color="#CCCCCC", lw=0.7)
    ax.axhline(0, color="#EEEEEE", lw=0.6)
    ax.axvline(0, color="#EEEEEE", lw=0.6)

    def arrow(v, color, label, lw=1.4, ls="-", off=(0, 0)):
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    linestyle=ls, shrinkA=0, shrinkB=0))
        ax.text(v[0] + off[0], v[1] + off[1], label, color=color, fontsize=7.6)

    arrow(aj, BLUE, "$a_j$", off=(0.03, -0.11))
    arrow(ap, GREY, "$a_p$", off=(0.02, 0.02))
    arrow(e, DARK, "$e_j^{*}$ (dual)", off=(0.02, 0.02))
    # projection of e onto a_p is zero -> dashed perpendicular
    ax.plot([e[0], e[0]], [0, e[1]], ls=":", color=DARK, lw=0.7)
    ax.text(e[0] * 0.5 + 0.05, e[1] * 0.5,
            "$a_p\\cdot e_j^{*}=0$", color=GREY, fontsize=6.9,
            rotation=90, va="center")
    ax.annotate("$a_j\\cdot e_j^{*}=1$", xy=(0.5, -0.02),
                xytext=(0.06, -0.34), fontsize=7.2, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.6))
    ax.text(0.60, 0.30, "gate $b\\,a_j$\nis a steeper V", color=ROSE,
            fontsize=7.0)
    ax.annotate("", xy=(0.62 * np.cos(0.35), 0.62 * np.sin(0.35)),
                xytext=(0.30, 0.10),
                arrowprops=dict(arrowstyle="->", color=ROSE, lw=0.8))
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.5, 0.95)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Dual vector: the zero-cost erase direction", fontsize=8.5,
                 color="#222222", pad=2)
    fig.tight_layout(pad=0.2)
    fig.savefig(outdir / "figDual.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="../revision/figures")
    args = ap.parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    fig_v(outdir)
    fig_rank(outdir)
    fig_dual(outdir)
    print("wrote", outdir / "figV.pdf", outdir / "figRank.pdf", outdir / "figDual.pdf")


if __name__ == "__main__":
    main()
