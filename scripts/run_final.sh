#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

NODE="zoo Mushroom NTU2012 cora pokec actor"; EDGE="iAF1260b iJR904 iSB619 iYO844"
for t in ${*:-node edge edge_rich}; do
  case $t in
    node)      for ds in $NODE; do node $ds 10 $(node_cfg $ds) --results_csv $RESULTS/node_${ds}.csv; done ;;
    edge)      for ds in $EDGE; do edge $ds 10 $(edge_cfg $ds) 100 --results_csv $RESULTS/edge_${ds}.csv; done ;;
    edge_rich) for ds in $EDGE; do edge $ds 10 $(rich_cfg $ds) 50 --x_dir $DATA/edge_rich_x \
                                     --results_csv $RESULTS/edge_rich_${ds}.csv; done ;;
    *) if [ -n "$(node_cfg $t)" ]; then node $t 10 $(node_cfg $t) --results_csv $RESULTS/node_${t}.csv
       elif [ -n "$(edge_cfg $t)" ]; then edge $t 10 $(edge_cfg $t) 100 --results_csv $RESULTS/edge_${t}.csv
       else echo "unknown target $t" >&2; exit 1; fi ;;
  esac
done
