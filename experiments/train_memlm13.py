#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the MemLM13 model: a hard linear-backbone LM with the trigger-gated
stable-key memory cell and a real selective-SSM density engine.

This is the immediate predecessor of ``train_memlm14.py`` (which upgrades the
backbone). It performs real held-out language modeling on wikitext-103
(train~120M / valid~251K / test~288K tokens) while keeping exact MQAR recall +
Δppl=0 zero pollution.

Training is two-phase (same protocol as v12):
  phase1: sparse MQAR recall (alpha=1, memory on)
  phase2: dense next-token (alpha=0, backbone only) -> honest wikitext PPL

Usage:
  python train_memlm13.py --steps 30000 --chunk 25000000 [--n_ssm 2 --N 16]
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
                    MemLM13, count_params, safe_step, eval_mqar, eval_wiki)


def run(args, dev):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    gen = make_mqar()
    rs = np.random.RandomState(0)
    KEYS = torch.from_numpy(rs.choice(VOCAB, 256, replace=False)).to(dev)
    model = MemLM13(VOCAB, args.dm, args.dk, args.dv, nl=args.nl, n_ssm=args.n_ssm,
                    N=args.N).to(dev)
    orth_init(model.cell, KEYS.cpu().numpy(), args.dk, dev)
    model.cell.key.requires_grad_(False)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), args.lr, weight_decay=args.wd)
    tr_tok = load_tokens("wikitext103_train")
    va_tok = load_tokens("wikitext103_valid")
    te_tok = load_tokens("wikitext103_test")
    print(f"=== memlm13 Dm={args.dm} nl={args.nl} n_ssm={args.n_ssm} N={args.N} "
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

    # phase2 dense on wikitext train chunk
    n_tr = min(args.chunk, len(tr_tok))
    mk, _ = wiki_loader(tr_tok, args.L, dev, seed=args.seed, offset=0, n=n_tr)
    log_every = max(1, int(args.steps / args.curve_pts))
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

    # final: valid/test dual readout + recall
    pf_v = eval_wiki(model, va_tok, args.L, dev, alpha=1.0, n_seq=args.eval_seq)
    pl_v = eval_wiki(model, va_tok, args.L, dev, alpha=0.0, n_seq=args.eval_seq)
    pf_t = eval_wiki(model, te_tok, args.L, dev, alpha=1.0, n_seq=args.eval_seq)
    pl_t = eval_wiki(model, te_tok, args.L, dev, alpha=0.0, n_seq=args.eval_seq)
    rec = eval_mqar(model, gen, [args.m], dev, bs=8)
    print(f"## FINAL valid_full={pf_v:.2f} valid_lm={pl_v:.2f} dV={pf_v-pl_v:.3f} "
          f"test_full={pf_t:.2f} test_lm={pl_t:.2f} dT={pf_t-pl_t:.3f} recall={rec}", flush=True)
    torch.save(model.state_dict(), f"memlm13_m{args.m}_s{args.seed}.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dm", type=int, default=1024)
    ap.add_argument("--dk", type=int, default=256); ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--nl", type=int, default=4); ap.add_argument("--n_ssm", type=int, default=2)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--m", type=int, default=256); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--chunk", type=int, default=25_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curve_pts", type=int, default=14)
    ap.add_argument("--eval_seq", type=int, default=256)
    args = ap.parse_args()
    run(args, "cuda" if torch.cuda.is_available() else "cpu")
