#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablation E2 — overload: with a 1024-key pool, compare orth-init vs random
frozen keys past load 1 (the load-one boundary).

Train phase1 at m=256 over a 1024-token key pool (each sample draws its own
256-subset), then evaluate recall at m in {256,512,768,1024}: beyond dk=256 strict
orthogonality is mathematically impossible, so this measures the degradation
regime of each key family.

  orth : QR-init rows of the 1024 pool keys, frozen  (approx-orthogonal; dk=256 caps 256 exact-orth)
  rand : default random Embedding rows, frozen
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

from dcgr3d import make_mqar, VOCAB, make_wmask, MemLM12, StableMemCell, orth_init, orth_penalty


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def make_gen_pool(poolsize, seed=0):
    rs = np.random.RandomState(seed)
    KEYS = rs.choice(VOCAB - 1, poolsize, replace=False)
    VALS = rs.choice(VOCAB - 1, poolsize, replace=False)
    PROBE = VOCAB - 1
    def gen(bs, m, d, dev):
        p = m - 1 - d
        ki = np.array([np.random.permutation(poolsize)[:m] for _ in range(bs)])
        vi = rs.randint(0, poolsize, (bs, m))
        keys = KEYS[ki]; vals = VALS[vi]
        L = 2 * m + 2
        seq = np.zeros((bs, L), dtype=np.int64); tgt = np.full((bs, L), -100, dtype=np.int64)
        seq[:, 0:2*m:2] = keys; seq[:, 1:2*m:2] = vals
        seq[:, 2*m] = PROBE; seq[:, 2*m+1] = keys[:, p]
        tgt[:, 1:2*m+1:2] = vals[:, :m]
        tgt[:, 2*m+1] = vals[:, p]
        wt = (tgt >= 0).astype(np.float32)
        return (torch.from_numpy(seq).to(dev), torch.from_numpy(tgt).to(dev),
                torch.from_numpy(wt).to(dev))
    return gen


def eval_ms(model, gen, ms, dev, bs_lo=16, bs_hi=4):
    model.eval(); model.alpha = 1.0
    with torch.no_grad():
        for m in ms:
            bs = bs_lo if m <= 256 else bs_hi
            r = {}
            for tag, d in [("d0", 0), ("dmid", (m-1)//2), ("dfar", m-1)]:
                accs = []
                for _ in range(2):
                    seq, tgt, _ = gen(bs, m, d, dev)
                    wm = make_wmask(tgt, seq.shape[1])
                    out = model(seq, wm)
                    pred = out[:, 2*m+1].argmax(-1)
                    accs.append((pred == tgt[:, 2*m+1]).float().mean().item())
                r[tag] = round(sum(accs)/len(accs), 3)
            print(f"## m={m} {r}", flush=True)
    model.train()


def key_geometry(model, KIDX):
    with torch.no_grad():
        K = model.cell.key(KIDX)
        Kn = F.normalize(K, dim=1)
        G = Kn @ Kn.t()
        off = G - torch.eye(G.shape[0], device=G.device)
        return dict(max_off=round(off.abs().max().item(), 4),
                    mean_abs_off=round(off.abs().mean().item(), 4))


def run_arm(arm, pool, m, n1, bs, Dm, nl, dk, dv, seed, dev, ms):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = make_gen_pool(pool, seed=0)
    rs = np.random.RandomState(0)
    KIDX = torch.from_numpy(rs.choice(VOCAB - 1, pool, replace=False)).to(dev)
    model = MemLM12(VOCAB, Dm, dk, dv, nl=nl, lam0=0.999, trigger=True, with_cell=True).to(dev)
    if arm == "orth":
        # QR of (pool, dk): columns orthonormal; for pool<=dk rows are exactly orthonormal,
        # for pool>dk rows are the best approx (rows pairwise near-orthogonal, rank dk).
        X = torch.randn(pool, dk, device=dev)
        Q, _ = torch.linalg.qr(X)
        with torch.no_grad():
            model.cell.key.weight.data[KIDX] = Q
        model.cell.key.requires_grad_(False)
    elif arm == "rand":
        model.cell.key.requires_grad_(False)
    print(f"=== arm={arm} pool={pool} m_train={m} params={count_params(model)} ===", flush=True)
    print(f"## arm={arm} KEYGRAM={key_geometry(model, KIDX)}", flush=True)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 3e-4)
    t0 = time.time()
    dset = list(range(m))
    for it in range(n1):
        d = dset[it % len(dset)]
        seq, tgt, wt = gen(bs, m, d, dev)
        opt.zero_grad()
        logits = model(seq, make_wmask(tgt, seq.shape[1]))
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none")
        loss = (ce * wt.reshape(-1)).sum() / wt.sum().clamp(min=1)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    print(f"## arm={arm} trained t={time.time()-t0:.0f}s", flush=True)
    eval_ms(model, gen, ms, dev)
    torch.save({"state": model.state_dict(), "arm": arm, "KIDX": KIDX.cpu()}, f"e2_{arm}.pt")
    print(f"## arm={arm} DONE saved e2_{arm}.pt t={time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=str, default="orth", choices=["orth", "rand"])
    ap.add_argument("--pool", type=int, default=1024)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n1", type=int, default=400)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--Dm", type=int, default=768)
    ap.add_argument("--nl", type=int, default=2)
    ap.add_argument("--dk", type=int, default=256)
    ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_arm(a.arm, a.pool, a.m, a.n1, a.bs, a.Dm, a.nl, a.dk, a.dv, a.seed, dev, [256, 512, 768, 1024])


if __name__ == "__main__":
    main()
