# The Cost of an Edit in a Fixed-State Associative Memory

Code and run records for the manuscript *The Cost of an Edit in a Fixed-State
Associative Memory* (Ziheng Zhou, working manuscript, September 2026).

The paper asks what governs the price of editing one association out of a
fixed-state outer-product memory. Its objects are a two-number cost --- the
**residual** (how much of the targeted binding remains) and the **locality**
(how far the surviving bindings move) --- and its results are: the residual
identity `residual = |1-g|` for *every* erase direction, a V-shaped trade-off
whose slope belongs to the direction rather than the code, the **dual-vector
condition** for a zero-cost erase, and the **Welch bound** that closes the
orthonormal route at realistic alphabet sizes.

## Layout

    dcgr3d/            the model and baselines (MemLM14 cell, delta rule, linear attention)
    experiments/       every script that produces a number in the paper
    data/revision/     the run records those scripts write (JSON / JSONL / log)
    tools/             small helpers for summarising runs

## Reproducing

**Analytic tables and the figures run on CPU** and need no checkpoints:

| Paper item | Script |
|---|---|
| Table I (erase V), Table II (seed dispersion), Table V (Welch floor) | `experiments/edit_locality.py` |
| Table VI min--max over `n=8` draws of the rule / `beta` / cliff tables | `experiments/edit_locality_seeds.py --n 8` |
| Table III (gate directions) and the cost multiplier | `experiments/edit_direction.py` |
| Figures 1--3 (V, dual vector, rank cliff) | `experiments/make_fig_v11.py --outdir figures` |

For example:

    python experiments/edit_locality.py
    python experiments/edit_locality_seeds.py --n 8
    python experiments/edit_direction.py
    python experiments/make_fig_v11.py --outdir figures

**The trained tables** (Table IX, Appendix A) need the trained checkpoints,
which are archived separately (~175--221 MB each). The scripts that consume
them are `experiments/benchmark_ear.py`,
`experiments/measure_gate_alignment.py` and
`experiments/gate_intervention.py`; `experiments/benchmark_ear.py` regenerates
the checkpoints from their recorded configuration.

## Requirements

    pip install -r requirements.txt

`torch` (CPU is enough for the analytic scripts), `numpy`, `matplotlib`.

## Data

`data/revision/` holds the run records behind the tables:
`locality_analytic.json` and `locality_direction.json` for the analytic sweeps,
and the per-flag JSONL files under `ear/`, `ear-fix/`, `fair/`, `dense/`,
`attribution/`, `tuning/` and `single_query/` for the trained arms. See
`data/revision/README.md`.

## License

MIT. See `LICENSE`.

## Citation

Ziheng Zhou. *The Cost of an Edit in a Fixed-State Associative Memory.* Working
manuscript, 2026.
