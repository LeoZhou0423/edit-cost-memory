#!/usr/bin/env python3
"""Compare the deletion-scored arm vs the un-scored (gated) arm from their run jsonl test events.

Reads:
  <rev>/ear-seeds/gated_s*_dk*.jsonl        (un-scored, arm='gated')
  <rev>/ear-delscored/del_s*_dk*.jsonl      (scored,     arm='delscored')

For every run's 'test' event it extracts the empirical deletion / recall metrics,
then reports per (arm, dk) means with [min, max] ranges.  The headline is the
deletion signal: q_del_token (=P(pred==NOISE), chance 1/50257) and stale_prob
(= mean prob mass on the stale token; high => deletion failed).

Optionally, if align_*.json files (learned gate g) are present next to the jsonl,
their g_mean is folded in as 'g_learned'.
"""
import os, json, glob, statistics as st

REV = os.path.dirname(os.path.abspath(__file__)) + "/../data/revision"
GATED_DIR = os.path.join(REV, "ear-seeds")
DEL_DIR = os.path.join(REV, "ear-delscored")

FIELDS = ["q_clean", "q_edited", "q_del_suppressed", "q_del_token",
          "stale_prob", "stale_rank_median", "erase_residual", "erase_locality"]


def load_runs(d, arm):
    rows = []
    for jf in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        if arm == "gated" and not os.path.basename(jf).startswith("gated_s"):
            continue
        if arm == "delscored" and not os.path.basename(jf).startswith("del_s"):
            continue
        with open(jf, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("event") != "test":
                    continue
                r = {"arm": arm, "dk": rec.get("dk"), "seed": rec.get("seed")}
                for k in FIELDS:
                    r[k] = rec.get(k, float("nan"))
                # learned gate from sibling align json if present
                base = os.path.basename(jf)[:-6]  # drop .jsonl
                align = os.path.join(d, "align_" + base + ".json")
                if os.path.exists(align):
                    try:
                        a = json.load(open(align))
                        r["g_learned"] = a.get("g_mean", float("nan"))
                    except Exception:
                        r["g_learned"] = float("nan")
                else:
                    r["g_learned"] = float("nan")
                rows.append(r)
    return rows


def agg(rows):
    out = {}
    for r in rows:
        key = (r["arm"], r["dk"])
        out.setdefault(key, []).append(r)
    res = []
    for (arm, dk), rs in sorted(out.items()):
        rec = {"arm": arm, "dk": dk, "n": len(rs)}
        for k in FIELDS + ["g_learned"]:
            vals = [x[k] for x in rs if x[k] == x[k]]  # drop nan
            if vals:
                rec[k] = (st.mean(vals), min(vals), max(vals))
            else:
                rec[k] = (float("nan"), float("nan"), float("nan"))
        res.append(rec)
    return res


def fmt(t):
    m, lo, hi = t
    if m != m:
        return "  -  "
    return f"{m:.4f} [{lo:.4f},{hi:.4f}]"


def main():
    rows = load_runs(GATED_DIR, "gated") + load_runs(DEL_DIR, "delscored")
    ag = agg(rows)
    # group by dk
    dks = sorted({r["dk"] for r in ag})
    arms = ["gated", "delscored"]
    cols = ["q_clean", "q_edited", "q_del_suppressed", "q_del_token",
            "stale_prob", "g_learned", "erase_residual", "erase_locality"]
    print(f"{'arm':10} {'dk':>4} {'n':>2}  " + "  ".join(f"{c:>16}" for c in cols))
    for dk in dks:
        for arm in arms:
            rec = next((r for r in ag if r["arm"] == arm and r["dk"] == dk), None)
            if not rec:
                print(f"{arm:10} {dk:>4}  -   (no data yet)")
                continue
            cells = "  ".join(f"{fmt(rec[c]):>16}" for c in cols)
            print(f"{arm:10} {dk:>4} {rec['n']:>2}  {cells}")
    # markdown
    md = ["| arm | dk | n | q_clean | q_edited | q_del_suppressed | q_del_token | stale_prob | g_learned | erase_residual | erase_locality |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for dk in dks:
        for arm in arms:
            rec = next((r for r in ag if r["arm"] == arm and r["dk"] == dk), None)
            if not rec:
                continue
            def mc(c):
                m, lo, hi = rec[c]
                return "—" if m != m else f"{m:.4f}"
            md.append("| " + " | ".join([arm, str(dk), str(rec["n"])] +
                       [mc(c) for c in cols]) + " |")
    with open(os.path.join(REV, "delscored_vs_gated.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\nwrote data/revision/delscored_vs_gated.md")


if __name__ == "__main__":
    main()
