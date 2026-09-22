"""Scatter and heatmap figures for the erase-gate results.

Reads whatever `align_*.json`, `inter_*.json` and `lossprobe_*.json` files are in
--dir and re-derives the panels, so the same script serves a three-run pilot
archive and the final multi-seed sweep:

  figGateAlign   trained gate vs dimension; the residual identity; the cost
                 multiplier of a channel gate.  Scatter, one point per seed.
  figGateSweep   the within-checkpoint gate sweep.  Heatmaps over
                 (forced gate) x (dk), with the as-trained row flagged.

Run from the repository root:

    python experiments/make_fig_gate.py --dir ../paper/data/revision/ear-seeds \
        --outdir ../paper/revision/figures
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from tools.palette import (BLUE, CYCLE, DARK, FAINT, GREY, INK, PINK, ROSE,
                           TEAL, apply_style, cmap_flat)

TAG_RE = re.compile(r"gated_s(\d+)_dk(\d+)")


def load_align(d):
    rows = []
    files = sorted(glob.glob(os.path.join(d, "align_*.json")) +
                   glob.glob(os.path.join(d, "gate_align_*.json")))
    for f in files:
        a = json.load(open(f, encoding="utf-8"))
        base = os.path.basename(f)[:-len(".json")]
        for pref in ("align_", "gate_align_"):
            if base.startswith(pref):
                base = base[len(pref):]
                break
        tag = base if base.startswith("gated_") else f"gated_s0_dk{a.get('dk')}"
        m = TAG_RE.search(tag)
        rows.append({
            "tag": tag, "dk": a.get("dk"), "step": a.get("step"),
            "seed": int(m.group(1)) if m else 0,
            "g": a.get("g_mean"), "g_min": a.get("g_min"), "g_max": a.get("g_max"),
            "resid": a.get("resid_gate"), "resid_pred": a.get("resid_gate_pred"),
            "mult": a.get("multiplier"), "slope_addr": a.get("slope_addr"),
            "mu": a.get("mu"), "L1_addr": a.get("L1_addr"),
            "L1_gate": a.get("L1_gate"),
        })
    return sorted(rows, key=lambda r: (r["dk"], r["seed"]))


def load_inter(d):
    """(dk, case) -> mean metrics over whatever checkpoints are present."""
    out = {}
    files = sorted(glob.glob(os.path.join(d, "inter_*.json")) +
                   glob.glob(os.path.join(d, "gate_intervention_*.json")))
    for f in files:
        a = json.load(open(f, encoding="utf-8"))
        dk = a.get("dk")
        for r in a.get("rows", []):
            key = (dk, r["case"])
            acc = out.setdefault(key, {"stale": [], "q_del": [], "q_clean": [],
                                       "gate": r.get("forced_gate")})
            acc["stale"].append(r["stale_prob"])
            acc["q_del"].append(r["q_del"])
            acc["q_clean"].append(r["q_clean"])
    return {k: {"stale": float(np.mean(v["stale"])),
                "stale_sd": float(np.std(v["stale"])),
                "q_del": float(np.mean(v["q_del"])),
                "q_clean": float(np.mean(v["q_clean"])),
                "n": len(v["stale"]), "gate": v["gate"]}
            for k, v in out.items()}


def load_lossprobe(d):
    """One entry per dk: the loss curve averaged over seeds.

    The probe is per checkpoint, so a multi-seed sweep produces several files per
    dk.  Averaging is what makes the panel readable; the across-seed spread is
    kept separately so a real seed effect cannot hide behind the mean.
    """
    bydk = {}
    for f in sorted(glob.glob(os.path.join(d, "lossprobe_*.json"))):
        a = json.load(open(f, encoding="utf-8"))
        e = bydk.setdefault(a.get("dk"), {"loss": {}, "trained": []})
        for r in a.get("rows", []):
            if r.get("gate") is None:
                continue
            e["loss"].setdefault(r["gate"], []).append(r["loss"])
        if a.get("loss_trained") is not None:
            e["trained"].append(a["loss_trained"])
    out = []
    for dk, e in sorted(bydk.items()):
        xs = sorted(e["loss"])
        mean = [float(np.mean(e["loss"][g])) for g in xs]
        sd = [float(np.std(e["loss"][g])) for g in xs]
        out.append({"dk": dk, "x": xs, "y": mean, "sd": sd,
                    "range": max(mean) - min(mean), "n": len(e["trained"]),
                    "trained": float(np.mean(e["trained"])) if e["trained"] else None})
    return out


CASES = ["as trained", "gate 0.00", "gate 0.25", "gate 0.50", "gate 0.75",
         "gate 1.00"]


def fig_align(rows, outdir):
    if not rows:
        return None
    dks = sorted({r["dk"] for r in rows})
    dcol = {dk: CYCLE[i % len(CYCLE)] for i, dk in enumerate(dks)}
    markers = ["o", "s", "^", "D", "v"][:max(1, len(dks))]
    dmk = {dk: markers[i % len(markers)] for i, dk in enumerate(dks)}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls=":", color=FAINT, lw=1.0, zorder=1)
    for r in rows:
        if r["g"] is None or r["resid"] is None:
            continue
        x = abs(1.0 - r["g"])
        ax.plot([x], [r["resid"]], marker=dmk[r["dk"]], ms=4.6,
                mfc=dcol[r["dk"]], mec=DARK, mew=0.6, ls="none", zorder=4,
                alpha=0.95)
    ax.set_xlabel(r"predicted $|1-g|$")
    ax.set_ylabel("measured residual")
    ax.set_title("Residual identity", pad=4)
    lo, hi = -0.03, 1.03
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.text(0.05, 0.93, f"$n={len(rows)}$ points\non the diagonal",
            transform=ax.transAxes, fontsize=6.6, color=GREY, va="top")

    def jitter(dk):
        """Spread the seeds of one dk symmetrically about its tick."""
        grp = [r for r in rows if r["dk"] == dk]
        n = len(grp)
        return {r["tag"]: (i - (n - 1) / 2.0) * 0.055 for i, r in enumerate(grp)}

    jit = {dk: jitter(dk) for dk in dks}

    ax = axes[1]
    for r in rows:
        if r["g"] is None:
            continue
        ax.plot([np.log2(r["dk"]) + jit[r["dk"]][r["tag"]]], [r["g"]],
                marker=dmk[r["dk"]], ms=4.6, mfc=dcol[r["dk"]], mec=DARK,
                mew=0.6, ls="none")
    med = {dk: np.median([r["g"] for r in rows if r["dk"] == dk]) for dk in dks}
    xs = np.log2(dks)
    ax.plot(xs, [med[dk] for dk in dks], "-", color=GREY, lw=0.9, zorder=1)
    for x, dk in zip(xs, dks):
        ax.plot([x], [med[dk]], "_", ms=9, color=INK, mew=1.2, zorder=5)
    ax.axhline(0.8815, ls=":", color=ROSE, lw=1.0)
    ax.text(0.03, 0.915, "initialisation", transform=ax.get_yaxis_transform(),
            fontsize=6.4, color=ROSE, va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{dk}" for dk in dks])
    ax.set_xlim(xs.min() - 0.45, xs.max() + 0.45)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"key dimension $d_k$")
    ax.set_ylabel(r"trained gate $g$")
    ax.set_title("Training pushes the gate down", pad=4)

    ax = axes[2]
    ax.axhspan(1.0, 3.4, color=PINK[0], zorder=0)
    ax.axhline(1.0, color=ROSE, lw=0.9)
    good = [r for r in rows if r["mult"] and r["mult"] < 30]
    for r in good:
        ax.plot([np.log2(r["dk"]) + jit[r["dk"]][r["tag"]]], [r["mult"]],
                marker=dmk[r["dk"]], ms=4.6, mfc=dcol[r["dk"]], mec=DARK,
                mew=0.6, ls="none")
    for x, dk in zip(xs, dks):
        v = [r["mult"] for r in good if r["dk"] == dk]
        if v:
            ax.plot([x], [np.median(v)], "_", ms=9, color=INK, mew=1.2)
    ax.set_yscale("log")
    ax.set_ylim(0.85, 3.4)
    ax.set_yticks([1, 1.5, 2, 3])
    ax.set_yticklabels(["1", "1.5", "2", "3"])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{dk}" for dk in dks])
    ax.set_xlim(xs.min() - 0.45, xs.max() + 0.45)
    ax.set_xlabel(r"key dimension $d_k$")
    ax.set_ylabel("cost multiplier")
    ax.set_title("Channel gate always costs more", pad=4)
    deg = [r["dk"] for r in rows if r["mult"] and r["mult"] >= 30]
    if deg:
        ax.text(0.97, 0.05, "excluded (degenerate\nslope): " +
                ", ".join(f"$d_k$={dk}" for dk in sorted(set(deg))),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2,
                color=GREY)

    handles = [Line2D([], [], marker=dmk[dk], ls="none", ms=4.6,
                      mfc=dcol[dk], mec=DARK, mew=0.6, label=f"$d_k={dk}$")
               for dk in dks]
    fig.legend(handles=handles, loc="lower center", ncol=len(dks),
               fontsize=6.8, handletextpad=0.3, columnspacing=1.1,
               bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.055, 1, 1), pad=0.35)
    out = outdir / "figGateAlign"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    return f"{out}.pdf"


def fig_sweep(inter, outdir):
    if not inter:
        return None
    dks = sorted({dk for dk, _ in inter})
    cases = [c for c in CASES if any((dk, c) in inter for dk in dks)]
    grid_s = np.full((len(cases), len(dks)), np.nan)
    grid_q = np.full((len(cases), len(dks)), np.nan)
    grid_d = np.full((len(cases), len(dks)), np.nan)
    for i, c in enumerate(cases):
        for j, dk in enumerate(dks):
            e = inter.get((dk, c))
            if e:
                grid_s[i, j] = e["stale"]
                grid_q[i, j] = e["q_clean"]
                grid_d[i, j] = e["q_del"]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    labels = [c.replace("gate ", "g = ").replace("as trained", "trained")
              for c in cases]

    def heat(ax, grid, cmap, title, cbar_label, fmt="{:.4f}", vmin=0.0,
             vmax=1.0, norm=None, dark_text=False):
        kw = {"cmap": cmap, "aspect": "auto", "origin": "upper"}
        if norm is not None:
            kw["norm"] = norm
        else:
            kw["vmin"], kw["vmax"] = vmin, vmax
        im = ax.imshow(grid, **kw)
        ax.set_xticks(range(len(dks)))
        ax.set_xticklabels([f"{dk}" for dk in dks])
        ax.set_yticks(range(len(cases)))
        ax.set_yticklabels(labels, fontsize=6.6)
        ax.set_xlabel(r"key dimension $d_k$")
        ax.set_title(title, pad=4)
        for s in ax.spines.values():
            s.set_visible(False)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if np.isnan(v):
                    continue
                c = INK if (dark_text or v < 0.55) else "#ffffff"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=6.3, color=c)
        # flag the as-trained row: it is the one the optimiser actually chose
        if cases and cases[0] == "as trained":
            ax.add_patch(plt.Rectangle((-0.5, -0.5), len(dks), 1, fill=False,
                                       ec=DARK, lw=1.1, zorder=6))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cbar_label, fontsize=6.4)
        cb.ax.tick_params(labelsize=6.0, width=0.6)
        cb.outline.set_linewidth(0.5)
        return im

    heat(axes[0], grid_s, cmap_flat(ROSE), "Stale value survives",
         "stale prob.", norm=LogNorm(vmin=0.0035, vmax=1.0))
    axes[0].text(0.5, -0.42, "chance 0.0039", transform=axes[0].transAxes,
                 ha="center", fontsize=6.2, color=ROSE)
    heat(axes[1], grid_d, cmap_flat(TEAL), "Suppression regained",
         r"$q_{\rm del}$")
    heat(axes[2], grid_q, cmap_flat(BLUE[4]), "Task accuracy unaffected",
         r"$q_{\rm clean}$", vmin=0.99, vmax=1.0, dark_text=True)
    fig.tight_layout(pad=0.35)
    out = outdir / "figGateSweep"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    return f"{out}.pdf"


def fig_loss(probes, g_of_dk, outdir):
    """Query-slot loss against the forced gate: does the objective want erasure?

    Deletion probes carry no target, so the only way the erase gate can move the
    training loss is by damaging the surviving items.  A flat curve therefore
    means the gate is not merely mis-learned -- it is unlearnable from this
    objective.
    """
    if not probes:
        return None
    n = len(probes)
    fig, axes = plt.subplots(1, n, figsize=(2.45 * n + 0.6, 2.2), squeeze=False)
    axes = axes[0]

    def _fmt(v, _pos):
        """Explicit tick text: matplotlib's 1e-5 offset label collides with the
        panel title, and a fixed decimal count flattens small values."""
        if v == 0:
            return "0"
        return f"{v:.2e}" if abs(v) < 1e-3 else f"{v:.4f}"

    for ax, p in zip(axes, probes):
        col = CYCLE[{32: 0, 64: 1, 256: 2}.get(p["dk"], 3) % len(CYCLE)]
        ys = list(p["y"]) + ([p["trained"]] if p["trained"] else [])
        lo, hi = min(ys), max(ys)
        span = max(hi - lo, 1e-12)
        ax.set_ylim(lo - 0.45 * span, hi + 0.45 * span)
        # The band is the *whole* variation of the curve: hairline thin means the
        # objective cannot tell the gate settings apart.
        ax.axhspan(lo, hi, color=PINK[0], lw=0, zorder=0)
        ax.plot(p["x"], p["y"], "-", color=col, lw=1.3, zorder=3)
        ax.plot(p["x"], p["y"], "o", ms=4.2, mfc="white", mec=DARK, mew=0.7,
                ls="none", zorder=4)
        if p["trained"] is not None and p["x"]:
            # x is the gate training actually left behind, not the initialisation:
            # only dk=256 happens to still sit at the initial value.  The trained
            # gate is per channel, so this point need not lie on the scalar sweep.
            xg = g_of_dk.get(p["dk"], 0.8815)
            ax.plot([xg], [p["trained"]], marker="*", ms=8.0, color=DARK,
                    ls="none", zorder=6)
        rel = p["range"] / abs(np.mean(ys)) if np.mean(ys) else float("nan")
        if p["n"] > 1 and max(p["sd"]) > 0:
            # How far the gate moves the loss, in units of the seed-to-seed spread.
            # Drawing the ribbon itself would swamp the curve; the ratio is the
            # quantity that decides whether the optimiser can see the gate at all.
            ax.text(0.04, 0.94, f"span / seed SD = {p['range'] / max(p['sd']):.2f}",
                    transform=ax.transAxes, fontsize=6.4, color=GREY, va="top")
        ax.set_xlabel("forced erase gate $g$")
        ax.set_title(f"$d_k={p['dk']}$  span {p['range']:.2e} ({rel:.1%})",
                     pad=7)
        ax.set_xlim(-0.02, 1.02)
        ax.tick_params(axis="both", pad=4.0, length=2.0, labelsize=7.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
        if ax is axes[0]:
            ax.set_ylabel("query-slot loss")
    fig.tight_layout(pad=0.35)
    out = outdir / "figGateLoss"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    return f"{out}.pdf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    apply_style()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_align(args.dir)
    inter = load_inter(args.dir)
    probes = load_lossprobe(args.dir)
    g_of_dk = {dk: float(np.mean([r["g"] for r in rows
                                  if r["dk"] == dk and r["g"] is not None]))
               for dk in {r["dk"] for r in rows}}
    print(f"align files : {len(rows)}  ({sorted({r['dk'] for r in rows})})")
    print(f"inter entries: {len(inter)}")
    print(f"loss probes : {len(probes)}")
    for f in (fig_align(rows, outdir), fig_sweep(inter, outdir),
              fig_loss(probes, g_of_dk, outdir)):
        if f:
            print("wrote", f)


if __name__ == "__main__":
    main()
