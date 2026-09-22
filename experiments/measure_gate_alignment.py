"""Measure the TRAINED erase gate's alignment, on the trained addressing code.

This is the mediator that the trained deletion result was missing.  The identity
of the paper is ``residual = |1 - g|`` with ``g = a_j . e_j``; the claim about the
gated cell was that its gate never saturates, so a delete leaves a residue that
nothing outranks.  Until now that ``g`` was quoted from the *initialisation*
(``sigmoid(2) = 0.881``), not measured on the trained weights.  This script
measures it.

For a stored key ``j`` the cell's erase branch computes

    b_j = sigmoid(gate([a_j, val(j)]))[:dk]        e_j = b_j * a_j

so, with ``a_j`` unit norm, ``g_j = sum_i b_ji a_ji^2`` and the read at the erased
address is left at ``|1 - g_j|`` of its former magnitude.  The script reports the
distribution of ``g_j``, the measured residual of that erase, and the cost
multiplier of the gate direction against the address direction on the *same*
trained code -- i.e. the slope ratio of Section 3.3, reproduced on trained
weights.

Run (on the training host, where the checkpoints live):

    python measure_gate_alignment.py \
        --ckpt ear-fix/gated_fixed_dk32.pt --out ear-fix/gate_align_dk32.json
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from dcgr3d.data import VOCAB, make_ear
from dcgr3d.model import MemLM14


def build(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    model = MemLM14(VOCAB, a["dm"], a["dk"], a["dv"], nl=a["nl"],
                    n_ssm=a["n_ssm"], N=a["state"], cell_kind=a["cell"])
    model.load_state_dict(ck["model"])
    model.to(device)
    model.eval()
    return model, a, int(ck.get("step", -1))


def costs(A, S, dirs):
    """(residual, locality, min alignment, mean alignment) for erase dirs."""
    m = A.shape[0]
    reads = A @ S
    resid, loc, align = [], [], []
    for j in range(m):
        e = dirs[j]
        g = float(A[j] @ e)
        Sp = S - torch.outer(e, A[j] @ S)
        r = float((A[j] @ Sp).norm() / (A[j] @ S).norm().clamp_min(1e-9))
        keep = torch.arange(m) != j
        base = reads[keep]
        now = A[keep] @ Sp
        l = float(((now - base).norm(dim=-1)
                   / base.norm(dim=-1).clamp_min(1e-9)).max())
        resid.append(r)
        loc.append(l)
        align.append(g)
    return (max(resid), max(loc), min(align), sum(align) / len(align))


@torch.no_grad()
def measure(model, a, device="cpu", n_batch=16, seed=7000):
    cell = model.cell
    gen = make_ear(seed=seed)
    rows = []
    for b in range(n_batch):
        seq, _, _, ops, meta = gen(1, a["m"], a["seq_len"], device, a["n_edit"])
        live = (meta["state"][0] != 2).nonzero().flatten()
        if len(live) < 3:
            continue
        key_ids = meta["stored_keys"][0][live]
        val_ids = meta["current"][0][live]
        A = torch.stack([cell._unit(cell.key(t)) for t in key_ids])
        Vv = torch.stack([cell.val(t) for t in val_ids])
        m = len(live)
        gram = A @ A.t()
        mu = float((gram - torch.eye(m, device=A.device)).abs().max())

        # the trained erase gate for every stored address
        trip = [cell._gates(key_ids[j].reshape(1), A[j].reshape(1, -1))
                for j in range(m)]
        gates = torch.stack([t[0].squeeze(0) for t in trip])       # erase, per key channel
        decay = torch.stack([t[1].squeeze(0) for t in trip])       # decay, per key channel
        write = torch.stack([t[2].squeeze(0) for t in trip])       # write, per value channel
        g_per_j = (gates * A.square()).sum(-1)          # g_j = sum_i b_ji a_ji^2

        S = A.t() @ Vv                                   # additive write, trained code
        ra, la, ga_min, ga_mean = costs(A, S, A)         # erase along the address
        rg, lg, gg_min, gg_mean = costs(A, S, gates * A)  # erase along the gate
        rows.append({
            "m": m, "mu": mu,
            "g_mean": float(g_per_j.mean()), "g_min": float(g_per_j.min()),
            "g_max": float(g_per_j.max()),
            "erase_gate_mean": float(gates.mean()),
            "erase_gate_min": float(gates.min()),
            "erase_gate_max": float(gates.max()),
            "decay_gate_mean": float(decay.mean()),
            "write_gate_mean": float(write.mean()),
            "resid_gate": rg, "resid_gate_pred": abs(1.0 - gg_min),
            "resid_addr": ra,
            "L1_gate": lg, "L1_addr": la,
            "slope_gate": lg / gg_min if gg_min > 1e-12 else float("nan"),
            "slope_addr": la / ga_min if ga_min > 1e-12 else float("nan"),
            "multiplier": (lg / gg_min) / (la / ga_min)
            if min(gg_min, ga_min) > 1e-12 else float("nan"),
        })
    if not rows:
        return None
    keys = rows[0].keys()
    agg = {"n_batch": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows]
        agg[k] = sum(vals) / len(vals)
        agg[k + "_max"] = max(vals)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, a, step = build(args.ckpt, device)
    res = measure(model, a, device, n_batch=args.batch)
    res = {"ckpt": args.ckpt, "step": step, "dk": a["dk"], "dv": a["dv"],
           "m": a["m"], "cell": a["cell"], **res}
    print(json.dumps(res, indent=1, sort_keys=True))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, sort_keys=True)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
