#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
ds=$1; MODES=${2:-"none no_agg no_additive no_features"}

for mode in $MODES; do
  out=$RESULTS/ablation_${ds}_${mode}.csv
  if [ -n "$(node_cfg $ds)" ]; then node $ds 10 $(node_cfg $ds) --ablation $mode --results_csv $out
  else edge $ds 10 $(edge_cfg $ds) 100 --ablation $mode --results_csv $out; fi
done
