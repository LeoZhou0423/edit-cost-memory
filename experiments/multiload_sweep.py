"""Multi-load check for the single-load limitation.

The analytic tables hard-code m=64.  This script repeats the core erase table
(evaluate() from edit_locality.py) at several loads so the central claim --- that
locality tracks coherence as L1/mu in [1.0, 1.5] --- can be stated over more than
one load.  Pure linear algebra, CPU only, no training.
"""
import argparse
import json

import torch

import edit_locality as el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/revision/locality_multiload.json")
    ap.add_argument("--loads", default="16,32,64,128")
    ap.add_argument("--dks", default="256,128,64,32")
    args = ap.parse_args()
    loads = [int(x) for x in args.loads.split(",")]
    dks = [int(x) for x in args.dks.split(",")]

    rows = []
    for m in loads:
        for dk in dks:
            # address erase along the (additive) write code, one draw, as in Table 1
            r = el.evaluate(m, dk, "additive", seed=0, erase_dir="address")
            rows.append({"m": m, "dk": dk, "mu": r["mu"], "welch": r["welch"],
                        "residual": r["residual"], "locality": r["locality"],
                        "ratio": r["locality"] / r["mu"] if r["mu"] > 1e-9 else None})
            print(f"m={m:>3} dk={dk:>3}  mu={r['mu']:.4f}  L1={r['locality']:.4f}"
                  f"  L1/mu={rows[-1]['ratio']:.3f}")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
