#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-measure generalization PPL + MQAR recall of saved models on held-out wiki
tokens (outside the training stream).

Training uses token[0:200000]; held-out uses token[200000:400000] (same seed).
Also compares a saved transformer checkpoint if present.

Usage:
  python eval_heldout.py --ckpt v12_model.pt --m 128 [--transformer_ckpt transformer.pt]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import math
import argparse

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch
import torch.nn.functional as F

from dcgr3d import make_mqar, dense_data, VOCAB, eval_mqar, MemLM12
from dcgr3d.baselines import TransformerLM


def ppl_nll(model, data, L, n=8):
    tot = 0.0; mt = 0
    with torch.no_grad():
        for _ in range(n):
            seq = data(8)
            model.alpha = 1.0
            o = model(seq)
            tot += F.cross_entropy(o[:, :-1].reshape(-1, VOCAB),
                                   seq[:, 1:].reshape(-1), reduction="sum").item()
            mt += 8 * (L - 1)
    if not math.isfinite(tot) or tot / mt >= 88 or tot < 0:
        return float("inf")
    return math.exp(tot / mt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="v12 checkpoint (state_dict or {'model':...})")
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--transformer_ckpt", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    L = 256
    gen = make_mqar()
    train_data = dense_data(200_000, L, dev, seed=123, offset=0)
    hold_data = dense_data(200_000, L, dev, seed=123, offset=200_000)

    model = MemLM12(VOCAB, 1024, 256, 128, nl=4, lam0=0.999,
                    trigger=True, with_cell=True).to(dev)
    sd = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
    model.eval()
    ppl_train = ppl_nll(model, train_data, L)
    ppl_hold = ppl_nll(model, hold_data, L)
    rec = eval_mqar(model, gen, [args.m], dev, bs=8)[f"m{args.m}"]
    print(f"v12: ppl_train={ppl_train:.3f} ppl_hold={ppl_hold:.3f} recall={rec}", flush=True)

    if args.transformer_ckpt and os.path.exists(args.transformer_ckpt):
        model = TransformerLM(VOCAB, 1024, 8, 4, window=None).to(dev)
        sd = torch.load(args.transformer_ckpt, map_location=dev)
        model.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
        model.eval()
        pt = ppl_nll(model, train_data, L)
        ph = ppl_nll(model, hold_data, L)
        print(f"transformer: ppl_train={pt:.3f} ppl_hold={ph:.3f}", flush=True)


if __name__ == "__main__":
    main()
