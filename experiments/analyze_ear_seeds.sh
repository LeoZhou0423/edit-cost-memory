#!/usr/bin/env bash
# Gate alignment + intervention for every checkpoint in ROOT.
#
# NOTE the PYTHONPATH.  `python experiments/foo.py` puts experiments/ -- not the
# cwd -- on sys.path[0], so `from dcgr3d.data import ...` inside the analysis
# scripts fails with ModuleNotFoundError unless the repo root is on the path
# explicitly.  This is the single most likely way to lose the analysis stage of a
# finished queue, so it is set here rather than relied on.
set -u
ROOT=${1:-/root/autodl-tmp/ear-seeds}
PY=${2:-/root/miniconda3/bin/python}
PREFIX=${3:-gated}          # gated = the paper's arm, del = the deletion-scored arm
REPO=/root/autodl-tmp/dcgr3d-revision
export PYTHONPATH=$REPO
cd "$REPO"

# the loss probe must be told whether deletion was scored, or it will measure a
# different objective from the one the checkpoint was trained under
if [ "$PREFIX" = "del" ]; then DELTGT=noise; else DELTGT=none; fi

for pt in "$ROOT"/${PREFIX}_s*_dk*.pt; do
  [ -f "$pt" ] || continue
  tag=$(basename "$pt" .pt)
  if [ ! -f "$ROOT/align_${tag}.json" ]; then
    "$PY" "$REPO/experiments/measure_gate_alignment.py" \
        --ckpt "$pt" --out "$ROOT/align_${tag}.json" >> "$ROOT/analysis.log" 2>&1 \
        || echo "FAIL align ${tag}"
  fi
  if [ ! -f "$ROOT/inter_${tag}.json" ]; then
    "$PY" "$REPO/experiments/gate_intervention.py" \
        --ckpt "$pt" --out "$ROOT/inter_${tag}.json" >> "$ROOT/analysis.log" 2>&1 \
        || echo "FAIL inter ${tag}"
  fi
  if [ ! -f "$ROOT/lossprobe_${tag}.json" ]; then
    "$PY" "$REPO/experiments/gate_loss_probe.py" \
        --ckpt "$pt" --del-target "$DELTGT" \
        --out "$ROOT/lossprobe_${tag}.json" >> "$ROOT/analysis.log" 2>&1 \
        || echo "FAIL lossprobe ${tag}"
  fi
done
echo "ANALYSIS_DONE"
