#!/usr/bin/env python3
"""Summarize JSONL artifacts emitted by benchmark_mqar_fair.py.

Reports the test record of every run next to the *candidate-set* chance level
``1/m``, because a model that has learned only "the answer is one of the m values
in the window" scores ``1/m`` while looking far above ``1/256``.

Usage:
    python tools/summarize_mqar.py "paper/data/revision/**/*.jsonl"
    python tools/summarize_mqar.py --curves "paper/data/revision/**/*.jsonl"
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict


def mean_sd(values):
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def load(paths):
    runs = {}
    for path in paths:
        records = [json.loads(line) for line in open(path, encoding="utf-8")
                   if line.strip()]
        if not records:
            continue
        start = next((r for r in records if r.get("event") == "start"), {})
        tag = start.get("tag") or os.path.basename(path)[:-6]
        tests = [r for r in records if r.get("event") == "test"]
        keys = ("task", "objective", "which", "m", "seq_len", "lr", "key_mode",
                "batch_size", "steps", "train_queries", "lr_schedule")
        runs[tag] = {
            "protocol": {k: start.get(k) for k in keys if k in start},
            "seeds": tests,
            "curve": [r for r in records if r.get("event") == "validation"],
            "path": path,
        }
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="JSONL files or glob patterns")
    parser.add_argument("--curves", action="store_true",
                        help="print validation curves instead of the test table")
    parser.add_argument("--metric", default="dfar",
                        help="metric to show in curves (default dfar)")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="recall level used for the steps-to-threshold table")
    args = parser.parse_args()
    paths = sorted({p for pattern in args.paths for p in glob.glob(pattern, recursive=True)})
    runs = load(paths)
    if not runs:
        raise SystemExit("No records found")

    if args.curves:
        for tag, run in sorted(runs.items()):
            p = run["protocol"]
            print(f"--- {tag}  {p.get('task')}/{p.get('objective')} "
                  f"m={p.get('m')} lr={p.get('lr')}")
            print(f"{'step':>8}{'loss':>10}{'d0':>8}{'dmid':>8}{'dfar':>8}"
                  f"{'dmq':>8}{'dall':>8}")
            for r in run["curve"]:
                row = f"{r['step']:>8}{r.get('loss', float('nan')):>10.4f}"
                for k in ("d0", "dmid", "dfar", "dmq", "dall"):
                    row += f"{r[k]:>8.4f}" if isinstance(r.get(k), float) else f"{'-':>8}"
                print(row)
        return

    groups = defaultdict(list)
    for tag, run in runs.items():
        groups[tag].append(run)

    print("run\tn\tchance(1/m)\td0\tdmid\tdfar\tdfar/(1/m)\tdmq\tdall")
    for tag, group in sorted(groups.items()):
        p = group[0]["protocol"]
        m = p.get("m") or 0
        chance = 1.0 / m if m else 0.0
        fields = {}
        for metric in ("d0", "dmid", "dfar", "dmq", "dall"):
            values = [r[metric] for run in group for r in run["seeds"]
                      if isinstance(r.get(metric), float)]
            fields[metric] = mean_sd(values) if values else None

        def fmt(metric):
            value = fields.get(metric)
            return f"{value[0]:.4f}" if value else "-"
        far = fields.get("dfar")
        ratio = f"{far[0] / chance:.2f}" if far and chance else "-"
        n = max((len(run["seeds"]) for run in group), default=0)
        print(f"{tag}\t{n}\t{chance:.4f}\t{fmt('d0')}\t{fmt('dmid')}\t{fmt('dfar')}"
              f"\t{ratio}\t{fmt('dmq')}\t{fmt('dall')}")

    print("\n# steps until the run first reaches the threshold on its own "
          "validation curve")
    print(f"# threshold = {args.threshold}; 'never' means the budget stopped "
          "before the phase transition")
    print("run\tscore_metric\tsteps@threshold\tfinal\tverdict")
    for tag, group in sorted(groups.items()):
        p = group[0]["protocol"]
        metric = "dall" if p.get("task") == "canonical" else "dfar"
        reached, finals = [], []
        for run in group:
            curve = [(r["step"], min(r["d0"], r["dmid"], r["dfar"]))
                     if metric == "dfar" else (r["step"], r["dall"])
                     for r in run["curve"]
                     if all(isinstance(r.get(k), float)
                            for k in ("d0", "dmid", "dfar"))]
            hit = next((step for step, value in curve if value >= args.threshold), None)
            reached.append(hit)
            if curve:
                finals.append(curve[-1][1])
        steps = [s for s in reached if s is not None]
        if not reached:
            continue
        if not steps:
            verdict = "never reached"
        elif len(steps) == len(reached):
            verdict = "all seeds"
        else:
            verdict = f"{len(steps)}/{len(reached)} seeds"
        mean = f"{sum(steps) // len(steps)}" if steps else "never"
        final = f"{sum(finals) / len(finals):.4f}" if finals else "-"
        print(f"{tag}\t{metric}\t{mean}\t{final}\t{verdict}")

    missing = [tag for tag, group in groups.items() if not group[0]["seeds"]]
    if missing:
        print("\n# runs without a test record (interrupted before test):")
        for tag in sorted(missing):
            print(f"#   {tag}  ({runs[tag]['path']})")


if __name__ == "__main__":
    main()
