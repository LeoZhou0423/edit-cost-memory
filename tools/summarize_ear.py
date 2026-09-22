"""Summarise edited-associative-recall runs into a manuscript table.

Reads every ``*.jsonl`` under the given roots, keeps the runs that reached a test
record, and prints one row per run with the trained code coherence, the edit
costs measured on the trained state, the same costs for the oracle dual write on
the identical addresses, and the three task metrics.

The comparison that matters is between ``erase_locality`` and ``coherence_mu``
(the prediction is that they agree, because the erase geometry is the code's),
and between ``erase_locality`` and ``oracle_dual_locality`` (the prediction is
that the dual, and only the dual, reaches zero).
"""
import argparse
import glob
import json
import os

FIELDS = ["coherence_mu", "erase_residual", "erase_locality", "locality_predicted",
          "oracle_dual_read_err", "oracle_dual_residual", "oracle_dual_locality",
          "q_clean", "q_edited", "q_del_suppressed", "stale_prob",
          "stale_rank_median", "n_clean", "n_edited", "n_deleted"]


def load(path):
    recs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    start = next((r for r in recs if r.get("event") == "start"), {})
    test = next((r for r in recs if r.get("event") == "test"), None)
    curve = [(r["step"], r.get("q_clean"), r.get("q_edited"))
             for r in recs if r.get("event") == "validation"]
    return start, test, curve


def steps_to_threshold(curve, threshold):
    for step, qc, qe in curve:
        if qc is not None and qe is not None and min(qc, qe) >= threshold:
            return step
    return None


def fmt(v, places=4):
    if isinstance(v, float):
        if v != v:
            return "nan"
        return f"{v:.{places}f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--curves", action="store_true")
    args = ap.parse_args()

    rows = []
    for root in args.roots:
        for path in sorted(glob.glob(os.path.join(root, "*.jsonl"))):
            start, test, curve = load(path)
            if test is None:
                continue
            rows.append((os.path.basename(path)[:-6], start, test, curve))

    if not rows:
        print("no completed runs found")
        return

    print("tag\tcell\tdk\torth\terase\tn\tsteps@%.2f\tmu\tresidual\tlocality"
          "\tdual_read\tdual_res\tdual_loc\tq_clean\tq_edited\tstale_rank" %
          args.threshold)
    for tag, start, test, curve in rows:
        hit = steps_to_threshold(curve, args.threshold)
        print("\t".join([
            tag, test.get("cell_kind", "?"), str(test.get("dk", "?")),
            "orth" if test.get("orth") else "learned",
            "off" if test.get("ablate_erase") else "on",
            str(test.get("n_clean", 0) + test.get("n_edited", 0)
                + test.get("n_deleted", 0)),
            str(hit) if hit else "never",
            fmt(test.get("coherence_mu")), fmt(test.get("erase_residual"), 3),
            fmt(test.get("erase_locality"), 3),
            fmt(test.get("oracle_dual_read_err"), 3),
            fmt(test.get("oracle_dual_residual"), 3),
            fmt(test.get("oracle_dual_locality"), 3),
            fmt(test.get("q_clean")), fmt(test.get("q_edited")),
            fmt(test.get("stale_rank_median"), 1),
        ]))

    if args.curves:
        for tag, start, test, curve in rows:
            print(f"\n# {tag}")
            print("#   step   q_clean  q_edited")
            for step, qc, qe in curve[::max(1, len(curve) // 10)]:
                print(f"# {step:>7}  {fmt(qc)}  {fmt(qe)}")


if __name__ == "__main__":
    main()
