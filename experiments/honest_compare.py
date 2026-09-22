#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Honest generalization comparison: on the same split (3.5M train / 0.5M
held-out), train a Transformer / Mamba backbone for dense LM and report true
held-out perplexity and MQAR recall — to answer objectively whether the linear
backbone's language-modeling generalization (~300) is an architecture weakness
or a corpus weakness (what an attention backbone achieves on the same corpus).

Usage:
  python honest_compare.py --which transformer|mamba --steps 12000 --seed 0
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import math
import argparse
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch
import torch.nn.functional as F
import numpy as np

from dcgr3d import make_mqar, dense_data, VOCAB, make_wmask, eval_mqar, eval_ppl
from dcgr3d.baselines import build, count_params


def run(args, dev):
    gen = make_mqar()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = build(args.which, 1024, args.seed).to(dev)
    nparam = count_params(model)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 1e-3, weight_decay=0.1)
    print(f"=== honest {args.which} Dm=1024 params={nparam} steps={args.steps} seed={args.seed} ===",
          flush=True)
    tr = dense_data(3_500_000, 256, dev, seed=args.seed, offset=0)
    hd = dense_data(500_000, 256, dev, seed=999, offset=3_500_000)
    t0 = time.time()
    log_every = max(1, args.steps // 12)
    for it in range(args.steps):
        seq = tr(16); opt.zero_grad()
        logits = model(seq, make_wmask(torch.full_like(seq, -100), seq.shape[1]))
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), seq[:, 1:].reshape(-1))
        ce.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (it + 1) % log_every == 0:
            ph = eval_ppl(model, hd, 256, dev, n=4)
            print(f"## {args.which} it={it+1}/{args.steps} ce={ce.item():.3f} "
                  f"held_out_ppl={ph:.2f} t={time.time()-t0:.0f}s", flush=True)
    ph = eval_ppl(model, hd, 256, dev)
    rec = eval_mqar(model, gen, [256], dev, bs=16)
    print(f"## FC({args.which}) held_out_ppl={ph:.2f} recall_m256={rec} "
          f"t={time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(), f"honest_{args.which}_s{args.seed}.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", type=str, required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args, "cuda" if torch.cuda.is_available() else "cpu")
