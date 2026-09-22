#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a saved MemLM13/MemLM14 checkpoint (valid/test PPL + MQAR recall)
without retraining. Reports the honest dual readout: memory on (``full``) vs
memory off (``lm``) — the two must agree (Δ=0) for the zero-interference claim.

Usage:
  python evaluate.py --ckpt memlm14_m256_s0.pt --model memlm14 [--dm 768 --nl 6 --n_ssm 3 --N 32]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch

from dcgr3d import (make_mqar, VOCAB, load_tokens, eval_mqar, eval_wiki,
                    MemLM13, MemLM14)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", choices=["memlm13", "memlm14"], default="memlm14")
    ap.add_argument("--dm", type=int, default=768)
    ap.add_argument("--dk", type=int, default=256); ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--nl", type=int, default=6); ap.add_argument("--n_ssm", type=int, default=3)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--L", type=int, default=256)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "memlm14":
        model = MemLM14(VOCAB, args.dm, args.dk, args.dv, nl=args.nl,
                        n_ssm=args.n_ssm, N=args.N).to(dev)
    else:
        model = MemLM13(VOCAB, args.dm, args.dk, args.dv, nl=args.nl,
                        n_ssm=args.n_ssm, N=args.N).to(dev)
    sd = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(sd)
    gen = make_mqar()
    va_tok = load_tokens("wikitext103_valid")
    te_tok = load_tokens("wikitext103_test")
    pf_v = eval_wiki(model, va_tok, args.L, dev, alpha=1.0, n_seq=2048)
    pl_v = eval_wiki(model, va_tok, args.L, dev, alpha=0.0, n_seq=2048)
    pf_t = eval_wiki(model, te_tok, args.L, dev, alpha=1.0, n_seq=2048)
    pl_t = eval_wiki(model, te_tok, args.L, dev, alpha=0.0, n_seq=2048)
    rec = eval_mqar(model, gen, [args.m], dev, bs=8)
    print(f"## EVAL {args.ckpt} model={args.model} dm={args.dm} m={args.m}")
    print(f"## valid_full={pf_v:.3f} valid_lm={pl_v:.3f} dV={pf_v-pl_v:.4f}")
    print(f"## test_full ={pf_t:.3f} test_lm ={pl_t:.3f} dT={pf_t-pl_t:.4f}")
    print(f"## recall_m{args.m}={rec}")


if __name__ == "__main__":
    main()
