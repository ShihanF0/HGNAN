#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
ds=$1; AGGS=${2:-"neighbor overall"}

best=-1
for agg in $AGGS; do for L in 3 5; do for hid in 64 256; do for dp in 0.0 0.5; do
  v=$(node $ds 3 $agg $L $hid $dp --log_every 100000 | val "Val Acc")
  echo "  $agg L=$L hidden=$hid dropout=$dp -> val acc ${v:-NA}"
  if [ -n "${v:-}" ] && awk "BEGIN{exit !($v > $best)}"; then best=$v; cfg="$agg $L $hid $dp"; fi
done; done; done; done

[ -n "${cfg:-}" ] || { echo "no result" >&2; exit 1; }
echo "selected: $cfg (val acc $best)"
node $ds 10 $cfg --results_csv $RESULTS/node_${ds}.csv
