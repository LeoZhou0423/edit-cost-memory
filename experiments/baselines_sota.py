#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Head-to-head against SSM / linear-attention SOTA baselines (matched scale).

Same protocol as the main model (identical make_mqar / dense_data / eval):
  - phase 1: MQAR masked loss (only value positions + probe), n1 steps
  - phase 2: dense wikitext next-token, steps steps
  - parameter count auto-matched to the full model (~106M)

Baselines (strongest forms, without the trigger-gate / memory-cell advantage):
  - mamba2 : selective SSM (Mamba-2 reference) + shared embedding head
  - linatt : infinitely-accumulating linear attention h += k⊗v (no forgetting)
  - transformer : causal attention (FlashAttention)

Usage:
  python baselines_sota.py --which mamba2 --m 64 --n1 1500 --steps 3000
  python baselines_sota.py --which linatt --m 256 ...
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import math
import time
import argparse

os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch
import torch.nn.functional as F
import numpy as np

from dcgr3d import make_mqar, dense_data, VOCAB, make_wmask, eval_mqar, eval_ppl
from dcgr3d.baselines import build, count_params

V = VOCAB


def run(args, dev):
    gen = make_mqar()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = build(args.which, args.Dm, args.seed).to(dev)
    nparam = count_params(model)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr)
    t0 = time.time()
    print(f"=== {args.which} Dm={args.Dm} m={args.m} params={nparam} seed={args.seed} ===",
          flush=True)

    cpath = f"curve_{args.which}_m{args.m}_s{args.seed}.csv"
    with open(cpath, "w") as f:
        f.write("step,phase,loss,recall\n")
    def log_curve(it, phase, loss, recall=""):
        with open(cpath, "a") as f:
            f.write(f"{it},{phase},{loss},{recall}\n")

    # ---- 阶段1: MQAR 掩码监督(与 main model phase1 逐位一致) ----
    dset = list(range(args.m))
    n1_log = max(1, args.n1 // 12)          # 每 1/12 采样一次 MQAR 召回
    for it in range(args.n1):
        d = dset[it % len(dset)]
        seq, tgt, wt = gen(args.bs, args.m, d, dev)
        opt.zero_grad()
        logits = model(seq, make_wmask(tgt, seq.shape[1]))
        ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="none")
        loss = (ce * wt.reshape(-1)).sum() / wt.sum().clamp(min=1)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        log_curve(it + 1, 1, f"{loss.item():.4f}")
        if (it + 1) % n1_log == 0:
            rec = eval_mqar(model, gen, [args.m], dev, bs=16)[str(args.m)]["dmid"]
            log_curve(it + 1, 1, f"{loss.item():.4f}", f"{rec}")
            print(f"## mqar{args.m} it={it+1}/{args.n1} loss={loss.item():.3f} "
                  f"recall_dmid={rec} t={time.time()-t0:.0f}s", flush=True)
    base = eval_mqar(model, gen, [args.m], dev, bs=16)
    print(f"## phase1 AFTER_MQAR m{args.m} recall={base} t={time.time()-t0:.0f}s", flush=True)

    # ---- 阶段2: dense LM(next-token) ----
    data = dense_data(200_000, args.L, dev)
    for it in range(args.steps):
        seq = data(args.bs); opt.zero_grad()
        ce = F.cross_entropy(model(seq)[:, :-1].reshape(-1, V), seq[:, 1:].reshape(-1))
        ce.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        log_curve(it + 1, 2, f"{ce.item():.4f}")
        if (it + 1) % max(1, args.steps//10) == 0:
            print(f"## dense it={it+1}/{args.steps} ce={ce.item():.3f} t={time.time()-t0:.0f}s",
                  flush=True)

    # ---- 最终读数：困惑度 + MQAR 召回 ----
    ppl = eval_ppl(model, data, args.L, dev, 4)
    mlist = [args.m, 128, 256] if args.m <= 128 else [args.m]
    rec = eval_mqar(model, gen, mlist, dev, bs=16)
    torch.save({"model": model.state_dict(), "nparam": nparam, "which": args.which,
                "mlist": mlist}, f"{args.which}.pt")
    print(f"## FINAL({args.which}) params={nparam} ppl={ppl:.2f} recall={rec} "
          f"wall={time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", type=str, default="mamba2",
                    choices=["mamba2", "linatt", "transformer"])
    ap.add_argument("--Dm", type=int, default=1024)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n1", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run(a, dev)


if __name__ == "__main__":
    main()
