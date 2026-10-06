#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
ds=$1

best=-1
if [ "${RICH:-0}" = "1" ]; then
  X="--x_dir $DATA/edge_rich_x"; PAT=50; HIDS="128 256"; SMS="2"; TAG=edge_rich
else
  X=""; PAT=100; HIDS="32 64 128"; SMS="1 2"; TAG=edge
fi
for dp in 0.0 0.5; do for hid in $HIDS; do for sm in $SMS; do
  v=$(edge $ds 3 $sm $hid $dp $PAT $X --log_every 100000 | val "Val AUROC")
  echo "  dropout=$dp hidden=$hid s_max=$sm -> val auroc ${v:-NA}"
  if [ -n "${v:-}" ] && awk "BEGIN{exit !($v > $best)}"; then best=$v; cfg="$sm $hid $dp"; fi
done; done; done

[ -n "${cfg:-}" ] || { echo "no result" >&2; exit 1; }
echo "selected: s_max hidden dropout = $cfg (val auroc $best)"
edge $ds 10 $cfg $PAT $X --results_csv $RESULTS/${TAG}_${ds}.csv
