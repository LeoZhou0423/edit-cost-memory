"""Force the erase gate on a trained checkpoint and watch deletion change.

Everything else is held fixed: the addressing code, the write gate, the decay,
the value embeddings, the backbone and the output head all stay exactly as
training left them.  Only the constant that multiplies the erase direction is
replaced, so the resulting change in deletion is attributable to the erase
direction and not to anything else that differs between runs.

This is the intervention the correlational version of the result could not make.
Table XIII of the manuscript measured the trained gate; it could not separate the
gate from the address dimension or from the training budget, because the three
rows differ in both.  Here one checkpoint is held still and the erase gate is
swept over it.

Interventions
-------------
  gate = trained      as the checkpoint left it
  gate = 0.0          no erase at all (the binding is untouched)
  gate = 0.25/0.5/... a partial erase
  gate = 1.0          erase along the address, the ungated direction
  alpha = 0           the memory readout is removed from the logits
  erase disabled      the ops tensor stops carrying delete/overwrite ops

Run (on the training host):

    python gate_intervention.py --ckpt ear-fix/gated_fixed_dk32.pt \
        --out ear-fix/gate_intervention_dk32.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from dcgr3d.data import VOCAB, make_ear
from dcgr3d.model import MemLM14


def build(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = dict(ck["args"])
    model = MemLM14(VOCAB, a["dm"], a["dk"], a["dv"], nl=a["nl"],
                    n_ssm=a["n_ssm"], N=a["state"], cell_kind=a["cell"])
    model.load_state_dict(ck["model"])
    model.to(device)
    model.eval()
    return model, a, int(ck.get("step", -1))


@torch.no_grad()
def eval_once(model, a, device, seed, batch=32, repeats=3, alpha=None,
              disable_erase=False):
    """Token-level EAR metrics, with the two controls the task was missing."""
    gen = make_ear(seed=seed)
    prev_alpha = getattr(model, "alpha", None)
    if alpha is not None and prev_alpha is not None:
        model.alpha = alpha
    hit = {"clean": 0, "edited": 0}
    tot = {"clean": 0, "edited": 0}
    del_hits, del_ranks, del_probs = [], [], []
    for _ in range(repeats):
        seq, tgt, wmask, ops, meta = gen(batch, a["m"], a["seq_len"], device,
                                         a["n_edit"])
        if disable_erase:
            ops = ops.clone()
            ops[ops == 2] = 0
        logits = model(seq, wmask, ops)
        pred = logits.argmax(-1)
        probs = logits.softmax(-1)
        for b in range(batch):
            for q in range(meta["query_slot"].shape[1]):
                slot = int(meta["query_slot"][b, q])
                pair = int(meta["query_pair"][b, q])
                kind = int(meta["query_kind"][b, q])
                if kind == 2:
                    stale = int(meta["current"][b, pair])
                    del_hits.append(int(int(pred[b, slot]) != stale))
                    del_ranks.append(float((logits[b, slot]
                                            > logits[b, slot, stale]).sum()))
                    del_probs.append(float(probs[b, slot, stale]))
                elif kind == 1:
                    tot["edited"] += 1
                    hit["edited"] += int(int(pred[b, slot]) == int(tgt[b, slot]))
                else:
                    tot["clean"] += 1
                    hit["clean"] += int(int(pred[b, slot]) == int(tgt[b, slot]))
    if prev_alpha is not None:
        model.alpha = prev_alpha
    out = {"q_clean": hit["clean"] / max(1, tot["clean"]),
           "q_edited": hit["edited"] / max(1, tot["edited"]),
           "q_del": (sum(del_hits) / len(del_hits)) if del_hits else float("nan"),
           "stale_prob": float(np.mean(del_probs)) if del_probs else float("nan"),
           "stale_rank": float(np.median(del_ranks)) if del_ranks else float("nan"),
           "n_del": len(del_hits)}
    return out


def summarise(runs):
    keys = ("q_clean", "q_edited", "q_del", "stale_prob", "stale_rank")
    out = {}
    for k in keys:
        v = [r[k] for r in runs]
        out[k] = float(np.mean(v))
        out[k + "_min"] = float(np.min(v))
        out[k + "_max"] = float(np.max(v))
    out["n_eval_seeds"] = len(runs)
    out["n_del"] = int(sum(r["n_del"] for r in runs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--seeds", default="5000,6000,7000,8000,9000")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]
    model, a, step = build(args.ckpt, device)
    cell = model.cell

    rows = []
    cases = [("as trained", dict(gate=None)),
             ("gate 0.00", dict(gate=0.0)),
             ("gate 0.25", dict(gate=0.25)),
             ("gate 0.50", dict(gate=0.5)),
             ("gate 0.75", dict(gate=0.75)),
             ("gate 1.00", dict(gate=1.0)),
             ("no memory readout", dict(gate=None, alpha=0.0)),
             ("erase disabled", dict(gate=None, disable_erase=True))]

    for name, kw in cases:
        cell.erase_override = kw.get("gate")
        runs = [eval_once(model, a, device, s, batch=args.batch,
                          repeats=args.repeats, alpha=kw.get("alpha"),
                          disable_erase=kw.get("disable_erase", False))
                for s in seeds]
        row = {"case": name, "forced_gate": kw.get("gate"),
               "alpha": kw.get("alpha"), "disable_erase": kw.get("disable_erase",
                                                                 False)}
        row.update(summarise(runs))
        rows.append(row)
        print(f"{name:>20}  q_clean={row['q_clean']:.4f}  "
              f"q_edit={row['q_edited']:.4f}  q_del={row['q_del']:.4f}  "
              f"stale_p={row['stale_prob']:.4f}  rank={row['stale_rank']:.1f}")
    cell.erase_override = None

    res = {"ckpt": args.ckpt, "step": step, "dk": a["dk"], "m": a["m"],
           "n_edit": a["n_edit"], "eval_seeds": seeds, "batch": args.batch,
           "repeats": args.repeats, "rows": rows}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, sort_keys=True)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
