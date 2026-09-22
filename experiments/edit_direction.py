"""Erase DIRECTION versus erase SCALE: does every gate lie on one V?

The V of the manuscript is derived by *scaling one fixed* erase direction: for
``e = c e_0``, ``residual = |1 - c g_0|`` and ``locality = c L_1``, so
eliminating the scale gives a line of slope ``L_1/g_0``.  A channel-wise gate
``e_j = b * a_j`` with a non-constant ``b`` is not a scaling of ``a_j``: it is a
different direction with a different ``g_0`` and a different ``L_1``, hence a
different slope.

Two things are measured here.

1.  **Each direction's own V.**  Scaling a fixed direction traces that
    direction's V: ``residual = |1 - c g_0|`` and ``locality = c L_1`` both hold,
    for every direction.  That is the universal half of the claim.

2.  **The cost multiplier at matched residual.**  At a given residual ``r``
    (pre-fold) the address direction costs ``s_addr (1 - r)`` with ``s_addr`` its
    slope; a gate direction costs ``s_gate (1 - r)``.  The ratio ``s_gate /
    s_addr`` is therefore exactly "how much more collateral this direction pays
    to reach the same residual", independent of where on the arm you sit.  A
    value below 1 would mean the gate beats scalar scaling; above 1 means it is
    dominated by it.

Run: python experiments/edit_direction.py [--out paper/data/revision/locality_direction.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edit_locality import evaluate  # noqa: E402

DK = 128
M = 64
SCALES = (0.25, 0.5, 1.0, 1.25, 1.5)
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)   # n = 8 code draws (was 3)


def gate_families():
    """(name, gate tensor builder).  A gate is (m, dk); row j is b_j."""
    g3 = torch.Generator().manual_seed(3)
    g4 = torch.Generator().manual_seed(4)
    g5 = torch.Generator().manual_seed(5)
    i = torch.arange(DK).float()
    return [
        ("all ones", torch.ones(M, DK)),
        ("uniform 0.5", 0.5 * torch.ones(M, DK)),
        ("half channels off", (torch.arange(DK) < DK // 2).float().repeat(M, 1)),
        ("random in [0.5,1]",
         (0.5 + 0.5 * torch.rand(DK, generator=g3)).repeat(M, 1)),
        ("random in [0,1]", torch.rand(DK, generator=g5).repeat(M, 1)),
        ("25% channels off",
         (torch.rand(DK, generator=g5) > 0.25).float().repeat(M, 1)),
        ("smooth sinusoid",
         (0.5 + 0.5 * torch.sin(2 * torch.pi * i / DK)).repeat(M, 1)),
        ("per-row random", torch.rand(M, DK, generator=g4)),
    ]


def sweep(dk, seed, name, gate_vec=None):
    kw = {"erase_dir": "gated", "gate_vec": gate_vec} if gate_vec is not None \
        else {"erase_dir": "address"}
    unit = evaluate(M, dk, "additive", seed=seed, gate_scale=1.0, **kw)
    g0, L1 = unit["alignment_g"], unit["locality"]
    rows = []
    for c in SCALES:
        r = evaluate(M, dk, "additive", seed=seed, gate_scale=c, **kw)
        rows.append({
            "direction": name, "dk": dk, "m": M, "seed": seed, "scale": c,
            "g0": g0, "L1": L1, "slope": L1 / g0,
            "g": r["alignment_g"],
            "residual": r["residual"], "pred_residual": abs(1.0 - c * g0),
            "locality": r["locality"], "pred_locality": c * L1,
            "mu": r["mu"],
        })
    return g0, L1, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows, summary = [], []
    print("=" * 100)
    print("Table: cost multiplier at matched residual.  m = 64, K = 256, "
          "dk = 128, additive write.")
    print("multiplier = (L_1/g_0) / (L_1/g_0 of the address direction at the "
          "same code draw).")
    print("1.00 = the gate is equivalent to scaling the address; >1 = it pays "
          "more collateral for the")
    print("same residual; <1 = it would beat scalar scaling.")
    print("=" * 100)
    print(f"{'direction':>22}" + "".join(f"{('s' + str(s)):>9}" for s in SEEDS)
          + f"{'mean':>9}{'median':>9}{'max':>9}")

    for name, vec in gate_families():
        mults = []
        for s in SEEDS:
            g0, L1, sub = sweep(DK, s, name, gate_vec=vec)
            ag0, aL1, _ = sweep(DK, s, "address")
            mult = (L1 / g0) / (aL1 / ag0)
            mults.append(mult)
            for r in sub:
                r["addr_slope"] = aL1 / ag0
                r["multiplier"] = mult
            rows += sub
        summary.append({"direction": name, "multipliers": mults})
        med = sorted(mults)[len(mults) // 2]
        print(f"{name:>22}" + "".join(f"{m:>9.3f}" for m in mults)
              + f"{sum(mults) / len(mults):>9.3f}{med:>9.3f}{max(mults):>9.3f}")
    print()

    allm = [m for s in summary for m in s["multipliers"]]
    below = [m for m in allm if m < 0.995]
    print(f"gate directions measured: {len(allm)} "
          f"({len(summary)} families x {len(SEEDS)} code draws)")
    print(f"multiplier  min {min(allm):.3f}   median "
          f"{sorted(allm)[len(allm) // 2]:.3f}   max {max(allm):.3f}")
    print(f"cases strictly below 1 (gate beats scalar scaling): {len(below)}")
    print()

    print("=" * 100)
    print("Check: within a fixed direction, scaling traces THAT direction's own V.")
    print("residual = |1 - c g0| and locality = c L1, with g0, L1 of that "
          "direction.  seed 0, dk = 128.")
    print("=" * 100)
    print(f"{'direction':>22}{'c':>7}{'resid':>9}{'|1-c g0|':>10}"
          f"{'loc':>9}{'c L1':>9}{'max dev':>9}")
    worst = 0.0
    for name, vec in gate_families():
        _, _, sub = sweep(DK, 0, name, gate_vec=vec)
        for r in sub:
            dev = max(abs(r["residual"] - r["pred_residual"]),
                      abs(r["locality"] - r["pred_locality"]))
            worst = max(worst, dev)
            if r["scale"] in (0.25, 1.0, 1.5):
                print(f"{name:>22}{r['scale']:>7.2f}{r['residual']:>9.4f}"
                      f"{r['pred_residual']:>10.4f}{r['locality']:>9.4f}"
                      f"{r['pred_locality']:>9.4f}{dev:>9.2e}")
    print(f"\nlargest deviation over all directions and scales: {worst:.2e}")
    print()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows, "summary": summary}, fh, indent=1,
                      sort_keys=True)
        print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
