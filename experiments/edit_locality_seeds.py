"""Multi-seed (n >= 5) dispersion for the analytical tables of the manuscript.

The geometry tables of ``edit_locality.py`` are one draw of the address code.
This driver repeats the three load-bearing ones over ``N`` independent code
draws and reports min--max (and mean), so a reader can judge every analytical
number against the sampling spread of the code itself.

  Table R   write rule x erase direction x dk        (manuscript Table tab:rules)
  Table B   the delta rule's step size beta          (manuscript Table tab:beta)
  Table C   the dual write inside and outside rank   (manuscript Table tab:cliff)

Run: python experiments/edit_locality_seeds.py [--n 8] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edit_locality import evaluate, address_code, LAM  # noqa: E402

POOL = 256


def rng(vals):
    return f"{min(vals):>8.4f} - {max(vals):<8.4f}"


def table_r(n):
    """write rule x erase direction x dk, m = 64, K = 256."""
    print("=" * 108)
    print("Table R.  Write rule and erase direction against address dimension, "
          f"m=64, K=256, n={n} draws.")
    print(f"{'dk':>4}{'rule':>10}{'erase':>9}{'mu':>18}{'read err':>20}"
          f"{'locality':>20}")
    rows = []
    for dk in (256, 128, 64, 32):
        for rule in ("additive", "delta", "dual"):
            for ed in ("address", "write"):
                out = {"dk": dk, "rule": rule, "erase": ed,
                       "mu": [], "read_err": [], "locality": []}
                for s in range(n):
                    try:
                        r = evaluate(64, dk, rule, seed=s, erase_dir=ed)
                    except Exception:                              # noqa: BLE001
                        out = None
                        break
                    out["mu"].append(r["mu"])
                    out["read_err"].append(r["read_err"])
                    out["locality"].append(r["locality"])
                if out is None:
                    print(f"{dk:>4}{rule:>10}{ed:>9}   failed")
                    continue
                rows.append(out)
                print(f"{dk:>4}{rule:>10}{ed:>9}{rng(out['mu']):>18}"
                      f"{rng(out['read_err']):>20}{rng(out['locality']):>20}")
        print()
    return rows


def table_b(n):
    """delta rule step size beta at an orthonormal (dk=256) and a coherent
    (dk=128) code."""
    print("=" * 108)
    print(f"Table B.  The delta rule's step size, n={n} draws.")
    print(f"{'dk':>4}{'beta':>7}{'read err':>20}{'locality':>20}")
    rows = []
    for dk in (256, 128):
        for beta in (1.0, 0.995, 0.98):
            re, loc = [], []
            for s in range(n):
                A = address_code(64, dk, seed=s)
                S = torch.zeros(dk, 64)
                for i in range(64):
                    k = A[i]
                    tgt = torch.zeros(64)
                    tgt[i] = 1.0
                    pred = k @ S
                    S = LAM * S + beta * torch.outer(k, tgt - LAM * pred)
                V = torch.randn(64, dk,
                                generator=torch.Generator().manual_seed(s + 1))
                St = S @ V          # S is (dk, m); V is (m, dk) -> (dk, dk)
                re.append(float((A @ St - V).norm(dim=-1).mean()
                                / V.norm(dim=-1).mean()))
                reads = A @ St
                l = 0.0
                for j in range(64):
                    Sp = St - torch.outer(A[j], A[j] @ St)
                    keep = torch.arange(64) != j
                    base = reads[keep]
                    now = A[keep] @ Sp
                    l = max(l, float(((now - base).norm(dim=-1)
                                      / base.norm(dim=-1).clamp_min(1e-9)).max()))
                loc.append(l)
            rows.append({"dk": dk, "beta": beta, "read_err": re, "locality": loc})
            print(f"{dk:>4}{beta:>7.3f}{rng(re):>20}{rng(loc):>20}")
        print()
    return rows


def table_c(n):
    """dual write inside and outside its rank."""
    print("=" * 108)
    print(f"Table C.  The exact dual write inside and outside its rank, "
          f"erase along the write code, n={n} draws.")
    print(f"{'dk':>4}{'m':>5}{'rule':>10}{'read err':>20}{'residual':>20}"
          f"{'locality':>20}")
    rows = []
    for dk in (64, 32, 16):
        for m in (dk // 2, dk, 2 * dk, 4 * dk):
            for rule in ("additive", "dual"):
                out = {"dk": dk, "m": m, "rule": rule,
                       "read_err": [], "residual": [], "locality": []}
                ok = True
                for s in range(n):
                    try:
                        r = evaluate(m, dk, rule, seed=s, erase_dir="write")
                    except Exception:                              # noqa: BLE001
                        ok = False
                        break
                    out["read_err"].append(r["read_err"])
                    out["residual"].append(r["residual"])
                    out["locality"].append(r["locality"])
                if not ok:
                    print(f"{dk:>4}{m:>5}{rule:>10}   failed")
                    continue
                rows.append(out)
                print(f"{dk:>4}{m:>5}{rule:>10}{rng(out['read_err']):>20}"
                      f"{rng(out['residual']):>20}{rng(out['locality']):>20}")
        print()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    payload = {"n": args.n,
               "table_r": table_r(args.n),
               "table_b": table_b(args.n),
               "table_c": table_c(args.n)}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
