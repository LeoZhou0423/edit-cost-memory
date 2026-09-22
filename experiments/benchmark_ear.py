"""Edited associative recall (EAR): does a fixed state support edits?

The task is canonical MQAR with an edit region inserted before the queries.  Each
edit is either an overwrite (rebind a stored key to a new value) or a delete
(remove a binding).  Three things are measured at the end of training:

1. **Task metrics.**  ``q_clean`` is recall on keys that were never edited,
   ``q_edit`` on keys that were overwritten, and ``q_del`` the fraction of
   deleted-key queries on which the stale value is no longer the model's argmax.
2. **State metrics.**  The addressing coherence ``mu = max |G - I|`` over the
   trained key rows, the residual left at an erased address, and the *locality*:
   the largest relative change an erase causes at an unedited address.
3. **Prediction.**  For an erase implemented as read-then-subtract, the read at
   an unedited address ``p`` changes by ``(a_p . a_j) (a_j^T S)``.  Since both
   reads are of the same order as ``|v|``, locality should track ``mu``.  The
   script prints the two side by side so the prediction can fail visibly.

Example
-------
python experiments/benchmark_ear.py --cell edit_add --dk 256 --orth \
    --m 8 --steps 3000 --seed 0
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcgr3d import (VOCAB, MemLM14, count_params, gpt_init, make_ear,
                    mqar_token_pools, orth_init)
from dcgr3d.memory import EditMemCell, GatedEditMemCell

NOISE = VOCAB - 1


def build_model(args, device):
    model = MemLM14(VOCAB, args.dm, args.dk, args.dv, nl=args.nl, n_ssm=args.n_ssm,
                    N=args.state, cell_kind=args.cell).to(device)
    if isinstance(model.cell, GatedEditMemCell):
        model.cell._gate_bias = tuple(float(v) for v in args.gate_bias.split(","))
    if args.emb_init == "gpt":
        model.apply(gpt_init)
        # gpt_init is generic and would flatten the gate's near-passthrough start,
        # so the gate initialisation is re-applied after it
        if isinstance(model.cell, GatedEditMemCell):
            model.cell.init_gates()
    if args.orth:
        key_ids, _ = mqar_token_pools(disjoint=True)
        orth_init(model.cell, key_ids, args.dk, device)
        model.cell.key.requires_grad_(False)
    return model


def learning_rate(step, args):
    warmup = max(1, int(args.steps * args.warmup_frac)) if args.warmup_frac else 0
    if warmup and step <= warmup:
        return args.lr * step / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, args.steps - warmup)))
    if args.lr_schedule == "cosine":
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.lr


@torch.no_grad()
def evaluate(model, args, device, data_seed, batch_size, repeats):
    """Token-level metrics on a fixed evaluation stream.

    A deleted key has no correct token, so it is scored by what happens to the
    *stale* value: whether it is still the argmax, and where it ranks.  The
    chance level for "stale value is not the argmax" is 255/256 = 0.996, which
    makes that fraction nearly uninformative on its own, so the rank and the
    probability of the stale value are reported alongside it.
    """
    gen = make_ear(seed=data_seed)
    model.eval()
    old = getattr(model, "alpha", None)
    if old is not None:
        model.alpha = 1.0
    hit = {"clean": 0, "edited": 0}
    total = {"clean": 0, "edited": 0}
    del_hits, del_ranks, del_probs = [], [], []
    for _ in range(repeats):
        seq, tgt, wmask, ops, meta = gen(batch_size, args.m, args.seq_len, device,
                                         args.n_edit)
        if args.ablate_erase:
            ops = ops.clone()
            ops[ops == 2] = 0
        logits = model(seq, wmask, ops)
        pred = logits.argmax(-1)
        probs = logits.softmax(-1)
        for b in range(batch_size):
            for q in range(meta["query_slot"].shape[1]):
                slot = int(meta["query_slot"][b, q])
                pair = int(meta["query_pair"][b, q])
                kind = int(meta["query_kind"][b, q])
                if kind == 2:
                    stale = int(meta["current"][b, pair])
                    del_hits.append(int(int(pred[b, slot]) != stale))
                    del_ranks.append(float((logits[b, slot] > logits[b, slot, stale])
                                           .sum()))
                    del_probs.append(float(probs[b, slot, stale]))
                elif kind == 1:
                    total["edited"] += 1
                    hit["edited"] += int(int(pred[b, slot]) == int(tgt[b, slot]))
                else:
                    total["clean"] += 1
                    hit["clean"] += int(int(pred[b, slot]) == int(tgt[b, slot]))
    if old is not None:
        model.alpha = old
    model.train()
    out = {}
    for key in ("clean", "edited"):
        out["q_" + key] = hit[key] / max(1, total[key])
    out["q_del_suppressed"] = (sum(del_hits) / len(del_hits)) if del_hits else float("nan")
    out["stale_rank_median"] = (float(np.median(del_ranks)) if del_ranks
                                else float("nan"))
    out["stale_prob"] = float(np.mean(del_probs)) if del_probs else float("nan")
    out["n_clean"] = total["clean"]
    out["n_edited"] = total["edited"]
    out["n_deleted"] = len(del_hits)
    return out


@torch.no_grad()
def state_metrics(model, args, device, data_seed=7000, batch=4):
    """Code coherence and edit locality read off the trained weights.

    Three things are measured on the same addresses and values, so they are
    directly comparable:

    - **the trained cell**, replayed through its own write and edit rules via
      ``state_after``;
    - **the oracle dual write**, ``S = A^T (A A^T)^{-1} V``, which makes the
      address code and the write code biorthogonal by construction.  Reads are
      then exact for ``m <= dk`` and erase locality is exactly zero, so this is
      the reference the delta rule is approximating;
    - **the prediction** ``locality <= mu``, where ``mu`` is the coherence of the
      trained address code.

    Only bindings that survive the edit stream are used: a deleted key has no
    read to compare against.
    """
    cell = model.cell
    if not isinstance(cell, (EditMemCell, GatedEditMemCell)):
        return {}
    gen = make_ear(seed=data_seed)
    rows = []
    for _ in range(batch):
        seq, _, _, ops, meta = gen(1, args.m, args.seq_len, device, args.n_edit)
        x = model.hidden(seq) if isinstance(cell, GatedEditMemCell) else None
        state = cell.state_after(seq, ops, x=x)[0]
        live = (meta["state"][0] != 2).nonzero().flatten()
        if len(live) < 2:
            continue
        key_ids = meta["stored_keys"][0][live]
        val_ids = meta["current"][0][live]
        A = torch.stack([cell._unit(cell.key(t)) for t in key_ids])
        Vv = torch.stack([cell.val(t) for t in val_ids])
        m = len(live)
        gram = A @ A.t()
        mu = float((gram - torch.eye(m, device=gram.device)).abs().max())

        if getattr(cell, "bilinear", False):
            def read(st, v):
                return torch.einsum('i,j,ijd->d', v, v, st)

            def subtract(st, v, rem):
                return st - torch.einsum('i,j,ijd->ijd', v, v, rem)
        else:
            def read(st, v):
                return v @ st

            def subtract(st, v, rem):
                return st - torch.einsum('i,d->id', v, rem)

        def probe(st):
            """(erase residual, locality) for erasing each binding in turn."""
            base = torch.stack([read(st, A[i]) for i in range(m)])
            resid, loc = [], []
            for j in range(m):
                after = subtract(st, A[j], read(st, A[j]))
                now = torch.stack([read(after, A[i]) for i in range(m)])
                keep = torch.arange(m, device=A.device) != j
                resid.append(float(now[j].norm() / base[j].norm().clamp_min(1e-9)))
                loc.append(float(((now[keep] - base[keep]).norm(dim=-1)
                                  / base[keep].norm(dim=-1).clamp_min(1e-9)).max()))
            return max(resid), max(loc)

        def read_err(st):
            got = torch.stack([read(st, A[i]) for i in range(m)])
            tgt = Vv
            return float((got - tgt).norm(dim=-1).mean()
                         / tgt.norm(dim=-1).mean().clamp_min(1e-9))

        trained_res, trained_loc = probe(state)

        # oracle dual write: S = A^T (A A^T)^{-1} V, i.e. write code B = (A A^T)^{-1} A
        G = gram.double()
        B = torch.linalg.solve(G, A.double()).to(A.dtype)
        dual_state = B.t() @ Vv
        dual_res, dual_loc = probe(dual_state)
        dual_read = read_err(dual_state)

        rows.append((mu, trained_res, trained_loc, m, dual_read, dual_res, dual_loc))
    if not rows:
        return {}
    arr = torch.tensor(rows, dtype=torch.float64).mean(0)
    return {"coherence_mu": float(arr[0]),
            "erase_residual": float(arr[1]),
            "erase_locality": float(arr[2]),
            "live_per_seq": float(arr[3]),
            "locality_predicted": float(arr[0]),
            "oracle_dual_read_err": float(arr[4]),
            "oracle_dual_residual": float(arr[5]),
            "oracle_dual_locality": float(arr[6])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="edit_add",
                    choices=["edit_add", "edit_delta", "edit_add_bi", "edit_gated"])
    ap.add_argument("--dm", type=int, default=512)
    ap.add_argument("--dk", type=int, default=256)
    ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--nl", type=int, default=6)
    ap.add_argument("--n-ssm", type=int, default=3)
    ap.add_argument("--state", type=int, default=32)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=0,
                    help="0 = 8m + 2*n_edit, matching the canonical query density")
    ap.add_argument("--n-edit", type=int, default=0, help="0 = m // 4")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-schedule", default="cosine", choices=["const", "cosine"])
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--emb-init", default="gpt", choices=["gpt", "default"])
    ap.add_argument("--orth", action="store_true",
                    help="frozen orthonormal addressing rows (needs dk >= 256)")
    ap.add_argument("--gate-bias", default="2,2,6",
                    help="gated cell only: initial biases of the erase, decay and "
                         "write gates, in that order.  sigmoid(2) = 0.881 leaves a "
                         "12%% erase residual; sigmoid(6) = 0.9975 is effectively a "
                         "pass-through.  Sweeping this separates 'gating is the "
                         "problem' from 'this gate initialisation is the problem'.")
    ap.add_argument("--ablate-erase", action="store_true",
                    help="drop the erase operations: writes land but nothing is removed")
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--eval-repeats", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=1000)
    ap.add_argument("--test-data-seed", type=int, default=5000)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    args.n_edit = args.n_edit or max(1, args.m // 4)
    args.seq_len = args.seq_len or 8 * args.m + 2 * args.n_edit

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model = build_model(args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.95))
    train_gen = make_ear(seed=args.data_seed)

    start = {"event": "start", "cell": args.cell, "orth": bool(args.orth),
             "ablate_erase": bool(args.ablate_erase), "m": args.m,
             "seq_len": args.seq_len, "n_edit": args.n_edit, "dk": args.dk,
             "dv": args.dv, "state_numbers": args.dk * args.dv
             * (args.dk if args.cell.endswith("bi") else 1),
             "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
             "lr_schedule": args.lr_schedule, "emb_init": args.emb_init,
             "seed": args.seed, "data_seed": args.data_seed,
             "test_data_seed": args.test_data_seed,
             "parameters": count_params(model)}
    with open(args.output, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(start, sort_keys=True) + "\n")
    print(json.dumps(start, sort_keys=True), flush=True)

    best, best_step = -1.0, 0
    started = time.time()
    for step in range(1, args.steps + 1):
        cur = learning_rate(step, args)
        for group in opt.param_groups:
            group["lr"] = cur
        seq, tgt, wmask, ops, meta = train_gen(args.batch_size, args.m, args.seq_len,
                                               device, args.n_edit)
        if args.ablate_erase:
            ops = ops.clone()
            ops[ops == 2] = 0
        opt.zero_grad(set_to_none=True)
        logits = model(seq, wmask, ops)
        mask = tgt >= 0
        loss = F.cross_entropy(logits[mask], tgt[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            rec = evaluate(model, args, device, args.data_seed + 1,
                           args.eval_batch_size, args.eval_repeats)
            score = min(rec["q_clean"], rec["q_edited"])
            if score > best:
                best, best_step = score, step
                if args.checkpoint:
                    torch.save({"model": model.state_dict(), "args": vars(args),
                                "step": step}, args.checkpoint)
            out = {"event": "validation", "cell": args.cell, "seed": args.seed,
                   "step": step, "lr": cur, "loss": float(loss.detach()),
                   "elapsed_seconds": time.time() - started, **rec}
            with open(args.output, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(out, sort_keys=True) + "\n")
            print(json.dumps(out, sort_keys=True), flush=True)

    if args.checkpoint and os.path.exists(args.checkpoint):
        sel = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(sel["model"])
    test = evaluate(model, args, device, args.test_data_seed,
                    args.eval_batch_size, args.eval_repeats)
    test.update(state_metrics(model, args, device))
    test.update({"event": "test", "cell": args.cell, "seed": args.seed,
                 "selected_step": best_step, "steps": args.steps,
                 "m": args.m, "n_edit": args.n_edit, "dk": args.dk, "orth": args.orth,
                 "ablate_erase": bool(args.ablate_erase), "cell_kind": args.cell})
    with open(args.output, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(test, sort_keys=True) + "\n")
    print(json.dumps(test, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
