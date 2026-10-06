#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
RAW=${RAW:-data/raw}
OUT=${OUT:-data/processed}

$PY build_node_data.py --datasets zoo NTU2012 cora --s_max 3 --raw_dir $RAW --out_dir $OUT/node
$PY build_node_data.py --datasets Mushroom pokec actor --s_max 2 --raw_dir $RAW --out_dir $OUT/node
$PY build_edge_data.py --datasets iAF1260b iJR904 iSB619 iYO844 --s_max 2 --raw_dir $RAW \
    --out_dir $OUT/edge --rich_x_dir $OUT/edge_rich_x
