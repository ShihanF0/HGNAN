# Interpretable Learning on Higher-Order Networks via Neural Additive Models

This repository contains the training code for **Interpretable Learning on Higher-Order Networks via Neural Additive Models**. It introduces HGNAN, a hypergraph neural additive network whose predictions decompose exactly into per-feature shape functions and distance-dependent structural contributions.

## Highlights

- Two tasks:
  - `node`: HGNAN-node for node classification on hypergraphs.
  - `edge`: HGNAN-edge for hyperedge prediction, used to recover missing reactions in genome-scale metabolic models (GEMs).
- Two structural aggregations: `overall` (learned weights over distance shells) and `neighbor` (sparse attention over one- and two-hop neighbors).
- Leak-free GEM benchmark: the reference graph is built from the training positives only.
- Ablations: `no_agg`, `no_additive`, and `no_features`.
- Supported datasets: Zoo, Mushroom, NTU2012, Cora, Pokec, Actor (node); iAF1260b, iJR904, iSB619, iYO844 (edge).

## Installation

Create a Python environment, then install the dependencies:

```bash
conda create -n hgnan python=3.10 -y
conda activate hgnan
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if the default wheel is not appropriate for your machine.

## Data Preparation

Place the raw data under `data/raw/<data_name>/`:

| Dataset | `--task` | `--data_name` | Raw files |
| --- | --- | --- | --- |
| Zoo, Mushroom, NTU2012 | `node` | `zoo`, `Mushroom`, `NTU2012` | `<name>.content`, `<name>.edges` (AllSet format) |
| Cora, Pokec, Actor | `node` | `cora`, `pokec`, `actor` | `features.pickle`, `labels.pickle`, `hypergraph.pickle` ([HyperUFG](https://github.com/kellysylvia77/HyperUFG)) |
| BiGG GEMs | `edge` | `iAF1260b`, `iJR904`, `iSB619`, `iYO844` | `<name>.mat` (stoichiometric matrix), `<name>.pt` (MACCS fingerprints of the metabolites) |

Then build the processed datasets:

```bash
bash scripts/build_data.sh
```

```text
data/processed/
├── node/<name>_smax<S>.pt
├── edge/<name>_smax2_split<seed>.pt    # one file per split
└── edge_rich_x/<name>_x.pt             # 504-d candidate features
```

## Training

The main entry point is [main.py](main.py). Each call trains 10 runs, one split per seed, and reports the mean and standard deviation.

Example:

```bash
python -u main.py \
  --task node \
  --data_name NTU2012 \
  --data_dir data/processed/node \
  --aggregation neighbor \
  --s_max 3 \
  --n_layers 3 \
  --hidden_channels 256 \
  --dropout 0.5 \
  --results_csv results/node_NTU2012.csv
```

```bash
python -u main.py \
  --task edge \
  --data_name iJR904 \
  --data_dir data/processed/edge \
  --aggregation overall \
  --s_max 2 \
  --n_layers 5 \
  --hidden_channels 128 \
  --patience 100 \
  --select val_auroc \
  --results_csv results/edge_iJR904.csv
```

For script wrappers:

```bash
bash scripts/run_final.sh                  # configurations reported in the paper
bash scripts/run_final.sh node             # or: edge | edge_rich | <data_name>
bash scripts/tune_node.sh NTU2012          # grid search on 3 seeds, then 10 seeds
bash scripts/tune_edge.sh iJR904
RICH=1 bash scripts/tune_edge.sh iJR904    # 504-d candidate features
bash scripts/run_ablation.sh NTU2012
```

The scripts read `PYTHON`, `DATA` and `RESULTS` from the environment. The selected configurations are listed in [scripts/common.sh](scripts/common.sh).

### Important Training Arguments

| Argument | Description |
| --- | --- |
| `--task` | `node` or `edge`. |
| `--data_name` / `--data_dir` | Dataset name and processed-data folder. |
| `--aggregation` | `overall` or `neighbor`. |
| `--s_max` | Largest `s` of the s-distances used by the model. |
| `--n_layers`, `--hidden_channels`, `--dropout` | Shape-function and distance-function MLPs. |
| `--select` | `val_loss` (node) or `val_auroc` (edge), used for early stopping and model selection. |
| `--ablation` | `none`, `no_agg`, `no_additive`, or `no_features`. |
| `--x_dir` | Folder with the 504-d candidate features. |

## Outputs

`--results_csv` writes one row per run:

```text
run, seed, val_acc, val_auroc, test_acc, test_auroc, test_auprc, test_f1, duration_s
```

`--save_dir` also saves the best checkpoint of every run as `<save_dir>/<data_name>/run<i>_seed<seed>.pt`.
