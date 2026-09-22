#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""needle-in-haystack: a single (key, value) bond buried in a long unrelated
context, queried by the ``PROBE`` token at the end.

The memory cell is O(1) content-addressed + trigger-gated, so it should recall
across arbitrary depth; a compressed-state baseline (Mamba-2) should degrade as
the needle is buried deeper. The memory layer only writes at the needle's value
position (same protocol as training).

Usage:
  python needle.py --ours_ckpt v12_model.pt [--mamba2_ckpt mamba2.pt] [--depth 8192]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import time
import argparse

os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch
import numpy as np

from dcgr3d import make_wmask, VOCAB, MemLM12
from dcgr3d.data import _data_dir
from dcgr3d.baselines import build

torch.set_grad_enabled(False)

KEYS0 = torch.from_numpy(np.random.RandomState(0).choice(VOCAB, 256, replace=False))
VALS0 = torch.from_numpy(np.random.RandomState(0).choice(VOCAB, 256, replace=False))
PROBE = VOCAB - 1


def run(which, model, rs, depth_list, wiki, dev, n_trials=200):
    """For each depth generate n_trials needle sequences and measure recall."""
    res = {}
    for D in depth_list:
        ok = 0
        for t in range(n_trials):
            ki = int(rs.randint(0, 256)); vi = int(rs.randint(0, 256))
            k = KEYS0[ki]; v = VALS0[vi]
            # 构造: H 填充 + (k,v) + (D-1) 填充 + PROBE + k
            H = 256                                   # 前面固定一段"无关背景"
            filler = wiki[rs.randint(0, len(wiki) - (H + D + 4), n_trials)[t]:]
            gap = D - 1
            total = H + 2 + gap + 2
            seq = np.empty(total, dtype=np.int64)
            seg = filler[:total]
            seq[:] = seg
            # 注入 needle: 索引 H..H+1 为 k,v ; 末尾 H+g+2=PROBE, H+g+3=k
            seq[H] = int(k); seq[H + 1] = int(v)
            seq[H + 2 + gap] = PROBE; seq[H + 3 + gap] = int(k)
            s = torch.from_numpy(seq[None].copy()).to(dev)
            tgt = torch.full((1, total), -100, dtype=torch.int64)
            tgt[0, H + 1] = v                                  # 只标记 needle 的 v
            wm = make_wmask(tgt, total)
            out = model(s, wm)
            pred = int(out[0, total - 1].argmax())
            ok += int(pred == v)
        res[D] = round(ok / n_trials, 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_ckpt", default="v12_model.pt")
    ap.add_argument("--mamba2_ckpt", default="")
    ap.add_argument("--wiki", default=None, help="path to wiki.npy (default: <data_dir>/wiki.npy)")
    ap.add_argument("--depths", type=str, default="128,256,512,1024,2048,4096,8192")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--out", default="needle_results.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rs = np.random.RandomState(7)
    depth_list = [int(x) for x in args.depths.split(",")]
    wiki_path = args.wiki or os.path.join(_data_dir(), "wiki.npy")
    wiki = np.load(wiki_path).astype(np.int64)
    out = {"meta": {"dev": dev, "depths": depth_list, "wiki": wiki_path}}

    # 我们的模型 (106M stable-key)
    ours = MemLM12(VOCAB, 1024, 256, 128, nl=4, lam0=0.999, trigger=True).to(dev)
    ck = torch.load(args.ours_ckpt, map_location=dev)
    ours.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    ours.eval()
    out["ours_106M"] = run("ours", ours, rs, depth_list, wiki, dev, n_trials=args.trials)
    print("ours recall_vs_depth:", out["ours_106M"], flush=True)

    # 可选 mamba2 基线
    if args.mamba2_ckpt and os.path.exists(args.mamba2_ckpt):
        mb = build("mamba2", 1024, 0).to(dev)
        mk = torch.load(args.mamba2_ckpt, map_location=dev)
        mb.load_state_dict(mk["model"] if isinstance(mk, dict) and "model" in mk else mk)
        mb.eval()
        out["mamba2_104M"] = run("mamba", mb, rs, depth_list, wiki, dev, n_trials=args.trials)
        print("mamba2 recall_vs_depth:", out["mamba2_104M"], flush=True)

    json.dump(out, open(args.out, "w"))
    print(f"## NEEDLE DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"wall={time.time()-t0:.0f}s", flush=True)
