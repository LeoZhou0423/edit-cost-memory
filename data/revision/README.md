# Revision MQAR evidence

Machine-readable artifacts backing the revised MQAR comparison. Every JSONL is
the raw stdout record stream of `repo/experiments/benchmark_mqar_fair.py`; the
matching `.log` file is the same stream plus the driver's own output.

## Layout

| Directory | Task / objective | Runs |
|---|---|---|
| `single_query/` | `loadone`, `--objective query` | `revision-runs-clean` (m=64, memory seeds 0-2, Transformer seeds 0-1), `transformer-tune` (m=64 learning-rate search), `transformer-capacity` (m=8/16/32) |
| `dense/` | `loadone`, `--objective multi`, and `canonical` | filled by the dense pilot, budget probe and canonical control |

Invalidated protocols live outside this tree, in
`archive/root_legacy/invalid_query_dilution/` (trivial copy targets) and
`archive/root_legacy/invalid_overlap/` (overlapping key/value pools). They must
not appear in tables.

## Reading a record

Each file starts with an `event: start` line carrying the full protocol
(`task`, `objective`, `m`, `seq_len`, `steps`, `batch_size`, `lr`,
`lr_schedule`, `warmup_frac`, `key_mode`, `dm`, parameter count, torch version),
then `event: validation` lines at every evaluation, and one closing
`event: test` line for the validation-selected checkpoint on the disjoint test
seed.

## Chance levels

Report recall against both references:

- `1/256` — pick any value token;
- `1/m` — pick one of the `m` values present in the window.

The second is the one that matters. A Transformer that has learned only "the
answer is one of the values I just saw" scores `1/m` while looking far above
vocabulary chance at large `m`. Under the single-query objective every measured
Transformer run sits at `1/m` for `m` in 8, 16, 32, 64, so that protocol cannot
distinguish "cannot bind" from "was not taught".

## Regenerating

```bash
cd repo
bash experiments/run_revision_suite.sh   /root/autodl-tmp/revision-runs-clean
bash experiments/run_revision_dense.sh   /root/autodl-tmp/dense-pilot
bash experiments/run_revision_budget.sh  /root/autodl-tmp/dense-budget
python tools/summarize_mqar.py "paper/data/revision/**/*.jsonl"
```

Checkpoints are written next to the JSONL files but are not committed: they are
220-320 MB each. Keep them on the remote data disk and record the run tags here.
