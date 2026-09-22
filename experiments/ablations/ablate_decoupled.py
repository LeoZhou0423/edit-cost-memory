#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablation E1 — is the decoupled (gated-off + frozen-key) protection necessary?

Arms (all share phase-1 MQAR at m, then dense next-token training for `steps`):
  base      : keys frozen (orthogonal init), dense training alpha=0   [paper default]
  nogate    : keys frozen,            dense training alpha=1           [memory left ON during dense]
  learnkeys : keys learnable,         dense training alpha=0           [keys not frozen during dense]
  both      : keys learnable,         dense training alpha=1           [no protection at all]

Prediction (paper): base keeps recall 1.000; nogate/learnkeys/both degrade it.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import math
import argparse
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch
import torch.nn.functional as F
import numpy as np

from dcgr3d import (make_mqar, dense_data, VOCAB, make_wmask, eval_mqar, eval_ppl,
                    MemLM12, StableMemCell, orth_init, orth_penalty)

ARMS = {"base": (1.0, True), "nogate": (1.0, False), "learnkeys": (0.0, True), "both": (0.0, False)}
# (dense_alpha_gate_on_bool_not, frozen)


def run_arm(arm, m, n1, steps, bs, L, Dm, nl, dk, dv, seed, dev, wiki, orth_lam):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = make_mqar()
    rs = np.random.RandomState(0)
    KEYS = torch.from_numpy(rs.choice(VOCAB, 256, replace=False)).to(dev)
    model = MemLM12(VOCAB, Dm, dk, dv, nl=nl, lam0=0.999, trigger=True, with_cell=True).to(dev)
    orth_init(model.cell, KEYS, dk, dev)
    frozen = (arm != "learnkeys" and arm != "both")
    model.cell.key.requires_grad_(not frozen)
    gate_on_dense = (arm == "nogate" or arm == "both")
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 3e-4)
    t0 = time.time()
    print(f"=== E1 arm={arm} m={m} n1={n1} steps={steps} Dm={Dm} nl={nl} frozen={frozen} "
          f"gate_on_dense={gate_on_dense} params={nparam} seed={seed} ===", flush=True)

    # phase 1: MQAR recall training (masked CE + orth penalty on learnable-key arms)
    dset = list(range(m))
    for it in range(n1):
        d = dset[it % len(dset)]
        seq, tgt, wt = gen(bs, m, d, dev)
        opt.zero_grad()
        logits = model(seq, make_wmask(tgt, seq.shape[1]))
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none")
        loss = (ce * wt.reshape(-1)).sum() / wt.sum().clamp(min=1)
        if not frozen:
            loss = loss + orth_penalty(model.cell, KEYS, orth_lam)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    rec0 = eval_mqar(model, gen, [m], dev, bs=8)
    print(f"## arm={arm} AFTER_MQAR recall={rec0} t={time.time()-t0:.0f}s", flush=True)

    # phase 2: dense next-token training (gate on for nogate/both; keys learnable for learnkeys/both)
    data = dense_data(200_000, L, dev, path=wiki)
    for it in range(steps):
        seq = data(bs); opt.zero_grad()
        model.alpha = 1.0 if gate_on_dense else 0.0
        ce = F.cross_entropy(model(seq)[:, :-1].reshape(-1, VOCAB), seq[:, 1:].reshape(-1))
        if not torch.isfinite(ce).all():
            continue
        ce.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (it + 1) % max(1, steps // 4) == 0:
            print(f"## arm={arm} dense it={it+1}/{steps} ce={ce.item():.3f} t={time.time()-t0:.0f}s", flush=True)

    # final: recall retention + perplexity full/lm
    model.alpha = 1.0
    rec1 = eval_mqar(model, gen, [m, min(256, m * 4)], dev, bs=8)
    ppl_full = eval_ppl(model, data, L, dev, n=4, alpha=1.0)
    ppl_lm = eval_ppl(model, data, L, dev, n=4, alpha=0.0)
    print(f"## E1 arm={arm} FINAL recall={rec1} ppl_full={ppl_full:.2f} ppl_lm={ppl_lm:.2f} "
          f"t={time.time()-t0:.0f}s", flush=True)
    torch.save({"state": model.state_dict(), "arm": arm}, f"e1_{arm}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=str, default="base", choices=list(ARMS))
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n1", type=int, default=1200)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--Dm", type=int, default=768)
    ap.add_argument("--nl", type=int, default=2)
    ap.add_argument("--dk", type=int, default=256)
    ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--orth_lam", type=float, default=1.0)
    ap.add_argument("--wiki", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_arm(a.arm, a.m, a.n1, a.steps, a.bs, a.L, a.Dm, a.nl, a.dk, a.dv, a.seed, dev, a.wiki, a.orth_lam)


if __name__ == "__main__":
    main()
