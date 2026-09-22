#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablation E1b (stress test) — does memory survive being activated during dense
training? Inject the reserved marker (PROBE = VOCAB-1) into a fraction of dense
sequences so the readout really writes/reads and receives dense gradients.

Arms:
  inj-frozen : keys frozen,      dense alpha=1, marker injected  -> memory live under dense grads
  inj-learn  : keys learnable,   dense alpha=1, marker injected  -> keys also receive dense grads

Question: does exact recall survive, or does joint training corrupt it?
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

PROBE = VOCAB - 1


def run_arm(arm, m, n1, steps, bs, L, Dm, nl, dk, dv, seed, dev, wiki, orth_lam, inject_frac):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = make_mqar()
    rs = np.random.RandomState(0)
    KEYS = torch.from_numpy(rs.choice(VOCAB, 256, replace=False)).to(dev)
    model = MemLM12(VOCAB, Dm, dk, dv, nl=nl, lam0=0.999, trigger=True, with_cell=True).to(dev)
    orth_init(model.cell, KEYS, dk, dev)
    learn_keys = (arm == "inj-learn")
    model.cell.key.requires_grad_(learn_keys)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 3e-4)
    t0 = time.time()
    print(f"=== E1b arm={arm} m={m} n1={n1} steps={steps} frozen={not learn_keys} "
          f"alpha_dense=1 inject_frac={inject_frac} params={nparam} seed={seed} ===", flush=True)

    # phase 1: MQAR recall
    dset = list(range(m))
    for it in range(n1):
        d = dset[it % len(dset)]
        seq, tgt, wt = gen(bs, m, d, dev)
        opt.zero_grad()
        logits = model(seq, make_wmask(tgt, seq.shape[1]))
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none")
        loss = (ce * wt.reshape(-1)).sum() / wt.sum().clamp(min=1)
        if learn_keys:
            loss = loss + orth_penalty(model.cell, KEYS, orth_lam)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    rec0 = eval_mqar(model, gen, [m], dev, bs=8)
    print(f"## arm={arm} AFTER_MQAR recall={rec0} t={time.time()-t0:.0f}s", flush=True)

    # phase 2: dense with marker injection (memory live, alpha=1)
    data = dense_data(200_000, L, dev, path=wiki)
    for it in range(steps):
        seq = data(bs).clone()
        if inject_frac > 0.0:
            # inject a PROBE at a random column of a random subset of rows
            n_inj = max(1, int(bs * inject_frac))
            ridx = torch.randint(0, bs, (n_inj,), device=dev)
            cidx = torch.randint(1, L - 1, (n_inj,), device=dev)
            seq[ridx, cidx] = PROBE
        opt.zero_grad()
        model.alpha = 1.0  # memory readout live (only where marker seen)
        ce = F.cross_entropy(model(seq)[:, :-1].reshape(-1, VOCAB), seq[:, 1:].reshape(-1))
        if not torch.isfinite(ce).all():
            continue
        ce.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (it + 1) % max(1, steps // 5) == 0:
            print(f"## arm={arm} dense it={it+1}/{steps} ce={ce.item():.3f} t={time.time()-t0:.0f}s", flush=True)

    model.alpha = 1.0
    rec1 = eval_mqar(model, gen, [m, min(256, m * 4)], dev, bs=8)
    ppl_full = eval_ppl(model, data, L, dev, n=4, alpha=1.0)
    ppl_lm = eval_ppl(model, data, L, dev, n=4, alpha=0.0)
    print(f"## E1b arm={arm} FINAL recall={rec1} ppl_full={ppl_full:.2f} ppl_lm={ppl_lm:.2f} "
          f"t={time.time()-t0:.0f}s", flush=True)
    torch.save({"state": model.state_dict(), "arm": arm}, f"e1b_{arm}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=str, default="inj-frozen", choices=["inj-frozen", "inj-learn"])
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n1", type=int, default=1200)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--Dm", type=int, default=768)
    ap.add_argument("--nl", type=int, default=2)
    ap.add_argument("--dk", type=int, default=256)
    ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--orth_lam", type=float, default=1.0)
    ap.add_argument("--inject_frac", type=float, default=0.25)
    ap.add_argument("--wiki", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_arm(a.arm, a.m, a.n1, a.steps, a.bs, a.L, a.Dm, a.nl, a.dk, a.dv, a.seed, dev, a.wiki, a.orth_lam, a.inject_frac)


if __name__ == "__main__":
    main()
