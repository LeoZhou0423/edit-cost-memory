#!/usr/bin/env bash
# Revival match: additive vs delta-rule vs second-order (bilinear) addressing,
# under the repaired protocol.
#
# Question. The 3D family was abandoned after dense training collapsed its MQAR
# recall, and the archived diagnosis was "objective conflict, not structure".  That
# diagnosis rests on a baseline that later turned out to be missing its residual
# (corrected_sweep.py) and on the single-query objective and overlapping key pools
# that experiment_protocol.md has since invalidated.  So the family's advantage is
# currently *unmeasured*, not disproven.  This queue measures it.
#
# Two axes, both analytic and both falsifiable:
#
#   A. matched state budget.  The linear cells use dk x dv state; the bilinear cell
#      uses dk x dk x dv, so at equal state it can only afford dk / dk^2.
#      ds = 32768: stable dk=256, delta dk=256, tensor3d dk=16
#      Prediction: the higher-order read loses, because more address dimensions
#      buy more orthogonal slots and orthogonality already drives the cross-talk
#      term to ~1e-7.
#
#   B. matched address dimension dk=16.  Here the bilinear cell has dk times more
#      state.
#      Prediction: the higher-order read wins, because the cross-talk term falls
#      from mu to mu^2 = 0.24^2 = 0.059.
#
# Together A and B answer the classic tensor-product-versus-compressed-binding
# trade in the fixed-state regime, at a fixed evaluation protocol.
#
# Evaluation: canonical layout, single-query-free (every query slot supervised),
# distance buckets not used; the metric is recall over all query slots plus
# steps-to-threshold on the validation curve.
#
# Usage: bash experiments/run_revision_revival.sh [ROOT] [PYTHON]

set -uo pipefail

ROOT="${1:-/root/autodl-tmp/revival}"
PY="${2:-/root/miniconda3/bin/python}"
mkdir -p "$ROOT"
cd "$(dirname "$0")/.."

SCHED="--lr-schedule cosine --warmup-frac 0.1"

run_one() {
  local tag="$1" m="$2" cell="$3" dk="$4" T="$5"
  if [ -f "$ROOT/${tag}.jsonl" ] && grep -q '"event": "test"' "$ROOT/${tag}.jsonl"; then
    echo "[$(date '+%F %T')] SKIP  ${tag}"
    return
  fi
  echo "[$(date '+%F %T')] START ${tag}"
  $PY experiments/benchmark_mqar_fair.py \
    --which memory --cell "$cell" --dk "$dk" --dv 128 \
    --task canonical --seq-len "$T" --m "$m" \
    --steps 2000 --batch-size 16 --eval-batch-size 16 --eval-repeats 8 \
    --eval-every 100 --dm 512 --nl 4 --n-ssm 2 --state 16 \
    --lr 3e-4 $SCHED --emb-init gpt --key-mode orth \
    --seed 0 --data-seed 1000 --test-data-seed 5000 --tag "$tag" \
    --output "$ROOT/${tag}.jsonl" --checkpoint "$ROOT/${tag}.pt" \
    > "$ROOT/${tag}.log" 2>&1 \
    && echo "[$(date '+%F %T')] DONE  ${tag}" \
    || echo "[$(date '+%F %T')] FAIL  ${tag} (see ${tag}.log)"
}

# Sequence length follows the Transformer reference runs (m=8 at T=64, m=64 at
# T=512), i.e. T = 8m, so the noise region holds 6m slots and the query density is
# 1/6.  Using T = 4m instead would roughly double the query density and make the
# task easier, which would break comparability with the reference runs.
for m in 8 16 32 64; do
  T=$((8 * m))
  # matched state budget (32768 numbers): 256 x 128 for the linear cells,
  # 16 x 16 x 128 for the bilinear cell
  run_one "a_state_stable_m${m}_dk256"   "$m" stable   256 "$T"
  run_one "a_state_delta_m${m}_dk256"    "$m" delta    256 "$T"
  # matched address dimension dk = 16 for all three cells (this run is shared by
  # both axes: it is the bilinear cell's only configuration)
  run_one "dk16_tensor3d_m${m}"          "$m" tensor3d 16  "$T"
  run_one "dk16_stable_m${m}"            "$m" stable   16  "$T"
  run_one "dk16_delta_m${m}"             "$m" delta    16  "$T"
done

echo "REVIVAL_FINISHED"
