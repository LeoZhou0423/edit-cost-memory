#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablation E1 — frozen-key geometry: orth vs random, on the stable-key harness.

Arms:
  orth        : QR-orthogonal init on the 256 key rows, keys frozen  (paper config)
  rand        : default nn.Embedding random keys, frozen, no orth init
  rand-learn  : random keys, learnable, orth_penalty drives them orthogonal during MQAR

Compares phase-1 MQAR exact recall at m=256 (load 1) and capacity m in {16,32,64,128}.
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

from dcgr3d import (make_mqar, VOCAB, make_wmask, eval_mqar, MemLM12,
                    StableMemCell, orth_init, orth_penalty)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def run_arm(arm, m, n1, bs, Dm, nl, dk, dv, seed, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = make_mqar()
    rs = np.random.RandomState(0)
    KEYS = torch.from_numpy(rs.choice(VOCAB, 256, replace=False)).to(dev)
    model = MemLM12(VOCAB, Dm, dk, dv, nl=nl, lam0=0.999, trigger=True, with_cell=True).to(dev)
    if arm == "orth":
        orth_init(model.cell, KEYS, dk, dev)          # QR-orthogonal rows
        model.cell.key.requires_grad_(False)
        orth_lam = 0.0                                 # frozen: penalty has no gradient anyway
        frozen = True
    elif arm == "rand":
        # default random Embedding rows stay as-is
        model.cell.key.requires_grad_(False)
        orth_lam = 0.0
        frozen = True
    elif arm == "rand-learn":
        model.cell.key.requires_grad_(True)            # keys learnable
        orth_lam = 1.0                                 # orth penalty pulls them orthogonal
        frozen = False
    nparam = count_params(model)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 3e-4)
    t0 = time.time()
    print(f"=== arm={arm} m={m} Dm={Dm} nl={nl} dk={dk} dv={dv} n1={n1} frozen={frozen} "
          f"params={nparam} seed={seed} ===", flush=True)
    dset = list(range(m))
    for it in range(n1):
        d = dset[it % len(dset)]
        seq, tgt, wt = gen(bs, m, d, dev)
        opt.zero_grad()
        logits = model(seq, make_wmask(tgt, seq.shape[1]))
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none")
        loss = (ce * wt.reshape(-1)).sum() / wt.sum().clamp(min=1)
        if arm == "rand-learn":
            loss = loss + orth_penalty(model.cell, KEYS, orth_lam)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    base = eval_mqar(model, gen, [m], dev, bs=16)
    print(f"## arm={arm} m={m} AFTER_MQAR recall={base} t={time.time()-t0:.0f}s", flush=True)
    # capacity view: evaluate at several store sizes with same frozen keys
    if m in (16, 32, 64, 128, 256):
        cap = eval_mqar(model, gen, [16, 32, 64, 128, 256], dev, bs=16)
        print(f"## arm={arm} CAPACITY={cap}", flush=True)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=str, default="orth", choices=["orth", "rand", "rand-learn"])
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n1", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--Dm", type=int, default=768)
    ap.add_argument("--nl", type=int, default=2)
    ap.add_argument("--dk", type=int, default=256)
    ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, flush=True)
    run_arm(a.arm, a.m, a.n1, a.bs, a.Dm, a.nl, a.dk, a.dv, a.seed, dev)


if __name__ == "__main__":
    main()
