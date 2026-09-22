#!/usr/bin/env python3
"""Convergence-aware MQAR benchmark for the revision.

Unlike the original uniform-short-schedule table, this runner records complete
learning curves and does not interpret a failed schedule as an architectural
ceiling. Run each family for every seed and learning rate, select hyperparameters
on validation examples, then evaluate the selected configuration once on a
separate test seed.

Two task layouts are available.

``--task loadone`` keeps the historical layout: ``m`` key-value pairs followed by
query blocks of ``[PROBE, key]``.  Under ``--objective query`` a `2m+2`-token
window carries a single supervised answer; ``--objective multi`` appends ``m``
query blocks and supervises every one.

``--task canonical`` uses the Zoology multi-query associative recall layout
(Arora et al., 2023): an ``m``-pair storage prefix, then a noise region in which
every stored key reappears once at a random slot.  Answers are supervised at all
query slots.  This is the layout published MQAR numbers are reported on, so it is
the control that separates "the family cannot bind" from "this layout and this
budget could not teach it".

Example
-------
python experiments/benchmark_mqar_fair.py --which transformer --dm 512 \
    --m 64 --steps 10000 --seed 0 --lr 3e-4 --objective multi
python experiments/benchmark_mqar_fair.py --which transformer --task canonical \
    --m 8 --seq-len 64 --steps 5000 --batch-size 64 --lr 3e-4 \
    --lr-schedule cosine --warmup-frac 0.1
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from dcgr3d import (VOCAB, MemLM14, count_params, gpt_init, make_mqar,
                    make_mqar_canonical, mqar_token_pools, mqar_write_mask,
                    orth_init)
from dcgr3d.baselines import build


def query_positions(m, queries):
    """Indices whose output is a retrieval answer in the load-one layout: the
    query-key slot of every query block.  The readout is content-addressed, so
    the answer sits at the same index as the key token that triggers it."""
    return [2 * m + 2 * q + 1 for q in range(queries)]


def learning_rate(step, args):
    """Linear warmup followed by constant, cosine or linear decay.

    Published MQAR runs use a warmup and a decay schedule, so a constant learning
    rate would under-train a baseline whose binding circuit only forms late."""
    warmup = max(1, int(args.steps * args.warmup_frac)) if args.warmup_frac else 0
    if warmup and step <= warmup:
        return args.lr * step / warmup
    progress = (step - warmup) / max(1, args.steps - warmup)
    progress = min(1.0, max(0.0, progress))
    if args.lr_schedule == "cosine":
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    if args.lr_schedule == "linear":
        return args.lr * (1.0 - progress)
    return args.lr


def supervised_loss(logits, target, args, queries):
    """Cross-entropy over retrieval answers only.

    The historical protocol also supervised every value-storage position, where
    the target equals the token already present at that position.  Those m
    trivial copy targets diluted the single retrieval target by a factor of m+1
    and let a baseline drive training loss down without learning retrieval, so
    they are gone.  What remains is one answer per sequence under
    ``--objective query`` and one per query slot under ``multi``/``canonical``.
    """
    if args.task == "canonical":
        mask = target >= 0
        return F.cross_entropy(logits[mask], target[mask])
    pos = torch.tensor(query_positions(args.m, queries), device=logits.device)
    return F.cross_entropy(logits.index_select(1, pos).transpose(1, 2),
                           target.index_select(1, pos))


def _recall_state(model):
    old_alpha = getattr(model, "alpha", None)
    if old_alpha is not None:
        model.alpha = 1.0
    return old_alpha


def _recall_restore(model, old_alpha):
    if old_alpha is not None:
        model.alpha = old_alpha
    model.train()


def recall_at_distances(model, data_seed, m, device, batch_size, repeats):
    """Load-one layout: single query per sequence, bucketed by storage distance."""
    gen = make_mqar(seed=data_seed, disjoint=True)
    model.eval()
    old_alpha = _recall_state(model)
    totals = {"d0": 0, "dmid": 0, "dfar": 0}
    denom = batch_size * repeats
    with torch.no_grad():
        for name, distance in (("d0", 0), ("dmid", (m - 1) // 2),
                               ("dfar", m - 1)):
            for _ in range(repeats):
                seq, target, _ = gen(batch_size, m, distance, device)
                logits = model(seq, mqar_write_mask(seq.shape[1], m, seq.device,
                                                    batch_size))
                pred = logits[:, 2 * m + 1].argmax(-1)
                totals[name] += int((pred == target[:, 2 * m + 1]).sum())
    _recall_restore(model, old_alpha)
    return {name: value / denom for name, value in totals.items()}


def recall_multi(model, data_seed, m, device, batch_size, repeats):
    """Load-one layout with m query blocks, uniformly drawn stored keys.

    The sequence is ``4m`` tokens here, so the batch is capped separately from
    the single-query pass: the readout materialises ``(batch, 4m, V)`` logits and
    a large vocabulary makes that the binding memory constraint."""
    gen = make_mqar(seed=data_seed, disjoint=True)
    model.eval()
    old_alpha = _recall_state(model)
    positions = query_positions(m, m)
    correct = 0
    with torch.no_grad():
        for _ in range(repeats):
            seq, target, _ = gen(batch_size, m, 0, device, queries=m)
            logits = model(seq, mqar_write_mask(seq.shape[1], m, seq.device,
                                                batch_size))
            for pos in positions:
                correct += int((logits[:, pos].argmax(-1) == target[:, pos]).sum())
    _recall_restore(model, old_alpha)
    return correct / (batch_size * repeats * m)


def recall_canonical(model, data_seed, m, seq_len, device, batch_size, repeats):
    """Canonical layout: accuracy over every query slot, plus the same slot split
    into thirds of the query region so early/middle/late queries are visible."""
    gen = make_mqar_canonical(seed=data_seed)
    model.eval()
    old_alpha = _recall_state(model)
    region_start = 2 * m
    region = max(1, seq_len - region_start)
    hits = [0, 0, 0]
    totals = [0, 0, 0]
    all_hits = all_total = 0
    with torch.no_grad():
        for _ in range(repeats):
            seq, target, _ = gen(batch_size, m, seq_len, device)
            logits = model(seq, mqar_write_mask(seq_len, m, seq.device, batch_size))
            mask = target >= 0
            pred = logits.argmax(-1)
            all_hits += int((pred[mask] == target[mask]).sum())
            all_total += int(mask.sum())
            rows, cols = mask.nonzero(as_tuple=True)
            slot = ((cols - region_start) * 3 // region).clamp(max=2)
            match = pred[rows, cols] == target[rows, cols]
            for k in range(3):
                sel = slot == k
                hits[k] += int(match[sel].sum())
                totals[k] += int(sel.sum())
    _recall_restore(model, old_alpha)
    return {"dall": all_hits / max(1, all_total),
            "d0": hits[0] / max(1, totals[0]),
            "dmid": hits[1] / max(1, totals[1]),
            "dfar": hits[2] / max(1, totals[2])}


def evaluate(model, args, data_seed, device):
    if args.task == "canonical":
        return recall_canonical(model, data_seed, args.m, args.seq_len, device,
                                args.eval_batch_size, args.eval_repeats)
    recall = recall_at_distances(model, data_seed, args.m, device,
                                 args.eval_batch_size, args.eval_repeats)
    if not args.skip_multi_eval:
        recall["dmq"] = recall_multi(model, data_seed, args.m, device,
                                     args.multi_eval_batch_size, args.eval_repeats)
    return recall


def build_model(args, device):
    emb_init = getattr(args, "emb_init", "gpt")
    if args.which == "memory":
        if getattr(args, "cell", "stable") == "tensor3d" and args.dk > 64:
            raise SystemExit("tensor3d state is dk x dk x dv; dk above 64 is not "
                             "affordable at these batch sizes. Lower --dk (and "
                             "note that this is the point of the comparison: the "
                             "bilinear address costs dk times more state).")
        model = MemLM14(VOCAB, args.dm, args.dk, args.dv, nl=args.nl,
                        n_ssm=args.n_ssm, N=args.state,
                        cell_kind=getattr(args, "cell", "stable")).to(device)
        # Same initialisation switch as the baselines, so the comparison is
        # symmetric. The published construction relied on the PyTorch defaults,
        # and its recall is carried by the cell algebra rather than by the head
        # scale, so both settings should reproduce it.
        if emb_init == "gpt":
            model.apply(gpt_init)
        # ``orth`` matches the published construction: the reserved key tokens
        # are given a QR-orthonormal basis and frozen.  This is an explicit
        # task-aligned algebraic prior, so ``learned`` (plain embedding init,
        # trainable) is the configuration that isolates how much of the gap the
        # prior accounts for.
        if getattr(args, "key_mode", "orth") == "orth":
            key_ids, _ = mqar_token_pools(disjoint=True)
            orth_init(model.cell, key_ids, args.dk, device)
            model.cell.key.requires_grad_(False)
    else:
        model = build(args.which, args.dm, args.seed, emb_init=emb_init,
                      depth=args.base_depth or None,
                      state_dim=args.base_state_dim or None).to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["memory", "mamba2", "linatt", "transformer"],
                        required=True)
    parser.add_argument("--cell", choices=["stable", "delta", "tensor3d"],
                        default="stable",
                        help="memory only: which cell sits in the same backbone and "
                             "readout interface. stable = additive outer product, "
                             "delta = residual (delta-rule) write, tensor3d = "
                             "second-order address with a trilinear read")
    parser.add_argument("--task", choices=["loadone", "canonical"], default="loadone")
    parser.add_argument("--objective", choices=["query", "multi"], default="query",
                        help="load-one only. query: one retrieval target per "
                             "sequence (historical protocol). multi: append "
                             "--train-queries query blocks and supervise every "
                             "one. canonical is always dense.")
    parser.add_argument("--train-queries", type=int, default=0,
                        help="query blocks per training sequence; 0 means m")
    parser.add_argument("--key-mode", choices=["orth", "learned"], default="orth",
                        help="memory only: orth = frozen QR-orthonormal keys "
                             "(published construction), learned = trainable "
                             "embedding keys (isolates the algebraic prior)")
    parser.add_argument("--emb-init", choices=["gpt", "default"], default="gpt",
                        help="gpt = GPT-2 style std=0.02 (standard, and what the "
                             "earlier baselines in this project used); default = "
                             "PyTorch defaults, which saturate the tied output "
                             "head at initialisation")
    parser.add_argument("--skip-multi-eval", action="store_true",
                        help="skip the extra m-query evaluation pass")
    parser.add_argument("--multi-eval-batch-size", type=int, default=16,
                        help="batch for the 4m-token multi-query pass, which "
                             "materialises (batch, 4m, V) logits")
    parser.add_argument("--m", type=int, default=64,
                        help="stored key-value pairs (and, for canonical, queries)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="canonical only: total sequence length")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-repeats", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", choices=["const", "cosine", "linear"],
                        default="const")
    parser.add_argument("--warmup-frac", type=float, default=0.0,
                        help="fraction of steps spent in linear warmup")
    parser.add_argument("--base-depth", type=int, default=0,
                        help="baselines only: number of blocks/state-mixer layers; "
                             "0 = the historical parameter-matching rule")
    parser.add_argument("--base-state-dim", type=int, default=0,
                        help="baselines only: recurrent state width (Mamba-2 "
                             "d_state, linear attention dk=dv); 0 = historical")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0, help="model/optimizer seed")
    parser.add_argument("--data-seed", type=int, default=1000)
    parser.add_argument("--test-data-seed", type=int, default=2000)
    parser.add_argument("--dm", type=int, default=512)
    parser.add_argument("--dk", type=int, default=256)
    parser.add_argument("--dv", type=int, default=128)
    parser.add_argument("--nl", type=int, default=4)
    parser.add_argument("--n-ssm", type=int, default=2)
    parser.add_argument("--state", type=int, default=16)
    parser.add_argument("--output", default="mqar_fair.jsonl")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tag", default="", help="free-form label kept in records")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Separate stateful generators prevent evaluation from consuming training
    # examples. The seed argument is part of the revised data protocol.
    train_gen = (make_mqar_canonical(seed=args.data_seed) if args.task == "canonical"
                 else make_mqar(seed=args.data_seed, disjoint=True))
    model = build_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    started = time.time()

    metadata = vars(args) | {
        "event": "start", "device": device, "parameters": count_params(model),
        "torch": torch.__version__,
    }
    with open(args.output, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, sort_keys=True) + "\n")

    best_score = -1.0
    best_step = 0
    train_queries = args.train_queries or args.m
    for step in range(1, args.steps + 1):
        distance = (step - 1) % args.m
        current_lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        if args.task == "canonical":
            seq, target, _ = train_gen(args.batch_size, args.m, args.seq_len, device)
        else:
            gen_queries = 1 if args.objective == "query" else train_queries
            seq, target, _ = train_gen(args.batch_size, args.m, distance, device,
                                       queries=gen_queries)
        optimizer.zero_grad(set_to_none=True)
        logits = model(seq, mqar_write_mask(seq.shape[1], args.m, seq.device,
                                            seq.shape[0]))
        loss = supervised_loss(logits, target, args, train_queries)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            recall = evaluate(model, args, args.data_seed + 1, device)
            score = min(value for key, value in recall.items() if key != "dmq")
            if score > best_score:
                best_score, best_step = score, step
                if args.checkpoint:
                    torch.save({"model": model.state_dict(), "args": vars(args),
                                "step": step, "validation": recall}, args.checkpoint)
            record = {
                "event": "validation", "which": args.which, "seed": args.seed,
                "task": args.task, "data_seed": args.data_seed, "step": step,
                "lr": current_lr, "loss": float(loss.detach()),
                "elapsed_seconds": time.time() - started, **recall,
            }
            with open(args.output, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)

    # Test the validation-selected checkpoint, never the arbitrary last step.
    if args.checkpoint and os.path.exists(args.checkpoint):
        selected = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(selected["model"])
    test_recall = evaluate(model, args, args.test_data_seed, device)
    final = {
        "event": "test", "which": args.which, "seed": args.seed, "task": args.task,
        "data_seed": args.test_data_seed, "step": args.steps,
        "best_validation_step": best_step, "best_validation_min": best_score,
        **test_recall,
    }
    with open(args.output, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(final, sort_keys=True) + "\n")
    print(json.dumps(final, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
