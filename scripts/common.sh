PY=${PYTHON:-python}
DATA=${DATA:-data/processed}
RESULTS=${RESULTS:-results}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$RESULTS"

smax () {
  case $1 in
    zoo|NTU2012) echo 3 ;;
    Mushroom|pokec|actor) echo 2 ;;
    cora) echo 1 ;;
    *) echo "unknown dataset $1" >&2; exit 1 ;;
  esac
}

node () {
  local ds=$1 runs=$2 agg=$3 L=$4 hid=$5 dp=$6; shift 6
  $PY -u main.py --task node --data_name $ds --data_dir $DATA/node --runs $runs \
      --aggregation $agg --s_max $(smax $ds) --n_layers $L --hidden_channels $hid --dropout $dp \
      --lr 0.001 --wd 0.0 --epochs 2000 --patience 50 --select val_loss "$@"
}

edge () {
  local ds=$1 runs=$2 sm=$3 hid=$4 dp=$5 pat=$6; shift 6
  $PY -u main.py --task edge --data_name $ds --data_dir $DATA/edge --runs $runs \
      --aggregation overall --s_max $sm --n_layers 5 --hidden_channels $hid --dropout $dp \
      --lr 0.001 --wd 0.0 --epochs 2000 --patience $pat --select val_auroc "$@"
}

node_cfg () {
  case $1 in
    zoo)      echo "neighbor 3 256 0.0" ;;
    Mushroom) echo "neighbor 3 64 0.0" ;;
    NTU2012)  echo "neighbor 3 256 0.5" ;;
    cora)     echo "overall 3 256 0.0" ;;
    pokec)    echo "neighbor 5 64 0.5" ;;
    actor)    echo "neighbor 5 256 0.0" ;;
  esac
}

edge_cfg () {
  case $1 in
    iAF1260b|iYO844) echo "2 128 0.5" ;;
    iJR904|iSB619)   echo "2 128 0.0" ;;
  esac
}

rich_cfg () {
  case $1 in
    iAF1260b|iJR904|iYO844) echo "2 256 0.5" ;;
    iSB619)                 echo "2 256 0.0" ;;
  esac
}

val () { grep -oE "^$1: +[0-9.]+" | grep -oE "[0-9.]+$" | head -1; }
