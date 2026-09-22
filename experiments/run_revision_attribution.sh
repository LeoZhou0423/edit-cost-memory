#!/usr/bin/env bash
# Attribution matrix for the two protocol repairs, all at m=8 where the control
# already converged (canonical, gpt init -> 0.9895 test recall in 8k steps).
#
#   layout / objective      emb_init=gpt            emb_init=default
#   canonical (dense)       0.9895  <- measured    run here
#   load-one, dense          run here              plateaued at 1/m
#   load-one, single query   run here              plateaued at 1/m
#
# Everything is held at batch 64, lr 3e-4, cosine with 10% warmup so the only
# differences are the layout, the objective and the initialisation.
#
# Usage: bash experiments/run_revision_attribution.sh [ROOT] [PYTHON]

set -uo pipefail

ROOT="${1:-/root/autodl-tmp/attribution}"
PY="${2:-/root/miniconda3/bin/python}"
mkdir -p "$ROOT"
cd "$(dirname "$0")/.."

decay="--lr-schedule cosine --warmup-frac 0.1"

run_one() {
  local model="$1" task="$2" m="$3" steps="$4" bs="$5" extra="$6" tag="$7"
  echo "[$(date '+%F %T')] START ${tag}"
  $PY experiments/benchmark_mqar_fair.py \
    --which "$model" --task "$task" --m "$m" --steps "$steps" \
    --batch-size "$bs" --eval-batch-size 64 --multi-eval-batch-size 16 \
    --eval-repeats 8 --eval-every 500 \
    --dm 512 --dk 256 --dv 128 --nl 4 --n-ssm 2 --state 16 \
    --lr 3e-4 --seed 0 --data-seed 1000 --test-data-seed 5000 --tag "$tag" \
    $extra \
    --output "$ROOT/${tag}.jsonl" --checkpoint "$ROOT/${tag}.pt" \
    > "$ROOT/${tag}.log" 2>&1 \
    && echo "[$(date '+%F %T')] DONE  ${tag}" \
    || echo "[$(date '+%F %T')] FAIL  ${tag} (see ${tag}.log)"
}

# init fix alone, dense objective, canonical layout
run_one transformer canonical 8 8000 64 "$decay --seq-len 64 --emb-init default" transformer_canon_m8_T64_defaultinit
# dense objective under the historical layout, fixed init
run_one transformer loadone   8 8000 64 "$decay --objective multi --emb-init gpt" transformer_loadone_m8_multi_gptinit
# sparse historical objective, fixed init, budget well past the phase transition
run_one transformer loadone   8 20000 64 "$decay --objective query --emb-init gpt" transformer_loadone_m8_query_gptinit
# control: canonical with the memory cell, same layout as the successful baseline
run_one memory      canonical 8 1500 64 "$decay --seq-len 64" memory_canon_m8_T64
run_one mamba2      canonical 8 6000 64 "$decay --seq-len 64" mamba2_canon_m8_T64
run_one linatt      canonical 8 6000 32 "$decay --seq-len 64" linatt_canon_m8_T64
# canonical at the load the literature reports on
run_one transformer canonical 64 12000 32 "$decay --seq-len 512" transformer_canon_m64_T512_b32

echo "ATTRIBUTION_FINISHED"
