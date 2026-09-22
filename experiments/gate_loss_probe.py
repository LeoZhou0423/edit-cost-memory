"""Is the training objective flat in the erase gate?

gate_intervention.py shows that forcing the erase gate to 1.0 restores deletion
while q_clean and q_edited stay at 1.0000.  That leaves two readings:

  (a) the optimiser *learned* a small gate because erasing costs accuracy -- the
      gate is doing its job; or
  (b) the objective simply does not constrain the gate, and the trained value is
      whatever initialisation plus drift produced.

Accuracy cannot tell them apart, because it is saturated at 1.0 across the whole
sweep.  The loss can.  This script sweeps the same gate values as
gate_intervention.py but reports the supervised cross-entropy on the query slots
-- which is exactly the training objective -- so the two readings separate:

  loss flat in g   -> (b) the gate is unconstrained; it cannot be learned
  loss rising in g -> (a) erasing costs real loss; the optimiser turns it off

Usage (on the training host, from the repository root):

    PYTHONPATH=. python experiments/gate_loss_probe.py \
        --ckpt ear-seeds/gated_s0_dk32.pt --out ear-seeds/lossprobe_s0_dk32.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gate_intervention as gi  # noqa: E402
from dcgr3d.data import make_ear  # noqa: E402


@torch.no_grad()
def eval_loss(model, a, device, seed, batch=32, repeats=3, del_target="none"):
    """Query-slot cross-entropy -- the training objective -- plus its accuracy.

    With ``del_target="none"`` (the paper's protocol) a deletion probe has no
    target and is excluded: we want the loss the optimiser actually sees, and
    that loss says nothing about the erased item.  With ``del_target="noise"``
    the deletion probes are supervised with the marker token and are included,
    which is the loss of the deletion-scored arm.
    """
    gen = make_ear(seed=seed, del_target=del_target)
    model.eval()
    old = getattr(model, "alpha", None)
    if old is not None:
        model.alpha = 1.0
    total_nll, total_n, hit, tot = 0.0, 0, 0, 0
    for _ in range(repeats):
        seq, tgt, wmask, ops, meta = gen(batch, a["m"], a["seq_len"], device,
                                         a["n_edit"])
        logits = model(seq, wmask, ops)
        logp = logits.log_softmax(-1)
        for b in range(batch):
            for q in range(meta["query_slot"].shape[1]):
                slot = int(meta["query_slot"][b, q])
                if int(meta["query_kind"][b, q]) == 2 and del_target != "noise":
                    continue
                t = int(tgt[b, slot])
                if t < 0:
                    continue
                total_nll += float(-logp[b, slot, t])
                total_n += 1
                hit += int(int(logits[b, slot].argmax()) == t)
                tot += 1
    if old is not None:
        model.alpha = old
    model.train()
    return {"loss": total_nll / max(1, total_n), "acc": hit / max(1, tot),
            "n": total_n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--seeds", default="5000,6000,7000,8000,9000")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--gates", default="0.0,0.25,0.5,0.75,0.9,1.0")
    ap.add_argument("--del-target", default="none", choices=["none", "noise"],
                    help="must match the arm the checkpoint was trained in")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]
    gates = [float(g) for g in args.gates.split(",")]
    model, a, step = gi.build(args.ckpt, device)
    cell = model.cell

    rows = []
    print(f"{'case':>10}  {'loss':>9}  {'acc':>7}   n")
    for gate in [None] + gates:
        cell.erase_override = gate
        runs = [eval_loss(model, a, device, s, batch=args.batch,
                          repeats=args.repeats, del_target=args.del_target)
                for s in seeds]
        loss = float(np.mean([r["loss"] for r in runs]))
        acc = float(np.mean([r["acc"] for r in runs]))
        lo = float(np.min([r["loss"] for r in runs]))
        hi = float(np.max([r["loss"] for r in runs]))
        rows.append({"gate": gate, "loss": loss, "loss_min": lo,
                     "loss_max": hi, "acc": acc, "n": int(runs[0]["n"])})
        print(f"{('trained' if gate is None else 'gate %.2f' % gate):>10}  "
              f"{loss:9.5f}  {acc:7.4f}   {runs[0]['n']}")
    cell.erase_override = None

    losses = [r["loss"] for r in rows if r["gate"] is not None]
    res = {"ckpt": args.ckpt, "step": step, "dk": a["dk"], "m": a["m"],
           "n_edit": a["n_edit"], "eval_seeds": seeds, "rows": rows,
           "loss_range": max(losses) - min(losses),
           "loss_trained": rows[0]["loss"]}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, sort_keys=True)
        print("wrote", args.out)
    print(f"loss range over g in [0,1]: {res['loss_range']:.6f}")


if __name__ == "__main__":
    main()
