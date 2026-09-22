#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the final MemLM14 model (stronger pure-linear backbone + trigger-gated
memory cell, honest held-out protocol).

v13 used a simplified ``SelectiveSSM(dm=512)`` backbone (honest held-out test
PPL=73.24). v14 pushes PPL into a more practical range while staying
attention-free and linear, upgrading only the backbone strength; the memory cell
(trigger gate / stable keys) is kept untouched so recall=1 / Δppl=0 is preserved.

Backbone upgrades (Mamba-2-style blocks, all linear/recurrent, no attention):
  1) larger capacity: bigger ``dm`` and a deeper SSM/MLP stack
  2) siLU gated dual branch (official Mamba structure)
  3) parallel scan: the block-closed-form scan (numerically stable, no NaN)
  4) full wikitext-103 (~120M tokens) trained to convergence

Protocol (honest held-out, identical to v13):
  - phase 1: sparse MQAR recall (alpha=1) -> recall
  - phase 2: dense next-token (alpha=0, backbone only) on the full corpus
  - final valid/test dual readout full(on) vs lm(off), Δ=+0 verifies zero pollution
  - appends a training-curve (step, ce, valid_ppl, size) CSV for plotting

Usage:
  python train_memlm14.py --dm 768 --nl 6 --n_ssm 3 --N 32 --steps 50000 --chunk 120000000
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

from dcgr3d import (make_mqar, make_wmask, VOCAB, load_tokens, wiki_loader,
                    StableMemCell, orth_init, orth_penalty, SelectiveSSM,
                    MemLM14, count_params, safe_step, eval_mqar, eval_wiki)


def run(args, dev):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    gen = make_mqar()
    rs = np.random.RandomState(0)
    KEYS = torch.from_numpy(rs.choice(VOCAB, 256, replace=False)).to(dev)
    model = MemLM14(VOCAB, args.dm, args.dk, args.dv, nl=args.nl, n_ssm=args.n_ssm,
                    N=args.N).to(dev)
    orth_init(model.cell, KEYS.cpu().numpy(), args.dk, dev)
    model.cell.key.requires_grad_(False)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), args.lr, weight_decay=args.wd)
    tr_tok = load_tokens("wikitext103_train")
    va_tok = load_tokens("wikitext103_valid")
    te_tok = load_tokens("wikitext103_test")
    print(f"=== memlm14 Dm={args.dm} nl={args.nl} n_ssm={args.n_ssm} N={args.N} "
          f"params={count_params(model)} steps={args.steps} chunk={args.chunk} seed={args.seed} ===",
          flush=True)
    t0 = time.time()

    # phase1 MQAR
    for it in range(args.m):
        d = it % args.m
        seq, tgt, wt = gen(args.bs, args.m, d, dev)
        opt.zero_grad()
        logits = model(seq, make_wmask(tgt, seq.shape[1]))
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none")
        loss = (ce * wt.reshape(-1)).sum() / wt.sum().clamp(min=1)
        loss = loss + orth_penalty(model.cell, KEYS)
        loss.backward(); safe_step(opt, model, 1.0)
    rec = eval_mqar(model, gen, [args.m], dev, bs=8)
    print(f"## phase1 MQAR recall m{args.m}={rec} t={time.time()-t0:.0f}s", flush=True)

    # phase2 dense on wikitext train chunk (full 120M)
    n_tr = min(args.chunk, len(tr_tok))
    mk, _ = wiki_loader(tr_tok, args.L, dev, seed=args.seed, offset=0, n=n_tr)
    log_every = max(1, int(args.steps / args.curve_pts))
    cpath = f"curve_memlm14_m{args.m}_s{args.seed}.csv"
    with open(cpath, "w") as f:
        f.write("step,ce,valid_ppl,size\n")
    for it in range(args.steps):
        seq = mk(args.bs); opt.zero_grad(); model.alpha = 0.0
        ce = F.cross_entropy(model(seq)[:, :-1].reshape(-1, VOCAB), seq[:, 1:].reshape(-1))
        if not torch.isfinite(ce).all():
            print(f"@@ dense it={it+1}/{args.steps} SKIP(nonfinite ce={ce.item()})", flush=True)
            continue
        ce.backward()
        safe_step(opt, model, 1.0)
        if (it + 1) % log_every == 0:
            pv = eval_wiki(model, va_tok, args.L, dev, alpha=0.0, n_seq=args.eval_seq)
            print(f"## dense it={it+1}/{args.steps} ce={ce.item():.3f} valid_ppl={pv:.2f} "
                  f"t={time.time()-t0:.0f}s", flush=True)
            with open(cpath, "a") as f:
                f.write(f"{it+1},{ce.item():.4f},{pv:.2f},{int(count_params(model))}\n")

    # final: valid/test dual readout + recall
    pf_v = eval_wiki(model, va_tok, args.L, dev, alpha=1.0, n_seq=args.eval_seq)
    pl_v = eval_wiki(model, va_tok, args.L, dev, alpha=0.0, n_seq=args.eval_seq)
    pf_t = eval_wiki(model, te_tok, args.L, dev, alpha=1.0, n_seq=args.eval_seq)
    pl_t = eval_wiki(model, te_tok, args.L, dev, alpha=0.0, n_seq=args.eval_seq)
    rec = eval_mqar(model, gen, [args.m], dev, bs=8)
    print(f"## FINAL valid_full={pf_v:.2f} valid_lm={pl_v:.2f} dV={pf_v-pl_v:.3f} "
          f"test_full={pf_t:.2f} test_lm={pl_t:.2f} dT={pf_t-pl_t:.3f} recall={rec}", flush=True)
    torch.save(model.state_dict(), f"memlm14_m{args.m}_s{args.seed}.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dm", type=int, default=768)
    ap.add_argument("--dk", type=int, default=256); ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--nl", type=int, default=6); ap.add_argument("--n_ssm", type=int, default=3)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--m", type=int, default=256); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--chunk", type=int, default=120_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curve_pts", type=int, default=20)
    ap.add_argument("--eval_seq", type=int, default=256)
    args = ap.parse_args()
    run(args, "cuda" if torch.cuda.is_available() else "cpu")
