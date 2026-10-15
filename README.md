# BattSABER

**Segmentation-Aware Basis Learning for Robust Battery Degradation Forecasting Under Heterogeneous Operating Conditions**

The source code for our proposed method will be released upon acceptance of the paper.

This repository currently provides the comparison baselines, processed datasets,
and experiment configurations for cross-condition battery capacity forecasting.

[中文说明](README.zh-CN.md)

## Models

**MLP · CNN · DLinear · LDNet · OmniTIEFormer · ModernTCN**

GRU is included as an optional baseline.

## Installation

Python 3.10 and PyTorch 2.7.1.

```bash
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CPU or CUDA environment.

## Dataset

The processed XJTU dataset is included in `data/xjtu_cccv36/`.

| Setting | Value |
| :--- | :--- |
| Batteries | 55 |
| Source conditions | 2C, 3C, R2.5, R3 |
| Target conditions | RW, Satellite |
| Train / validation / test | 30 / 9 / 16 batteries |
| Input | 64 cycles × 37 channels |
| Output | Capacity for the next 20 cycles |

Inputs contain 36 CC/CV features and historical capacity. The feature definitions
are in `data/feature_manifest.csv`; the battery split is in `configs/split.json`.

## Quick start

```bash
# Check the data and model interfaces
python run.py --action audit --device cpu

# Quick training test
python run.py --action smoke --device cpu

# Train the six baselines with the selected configurations
python run.py --action train --device cuda --resume
```

Use `--device cpu` without a GPU. Select a subset of models or seeds with:

```bash
python run.py --action train --methods MLP ModernTCN --seeds 1 --device cuda --output outputs/subset
```

To run GRU:

```bash
python run.py --action train --methods GRU --device cuda --output outputs/gru
```

## Training and evaluation

Default training uses batch size 256, Adam, MSE loss, up to 200 epochs and
validation-based early stopping with patience 10. Model seeds are 1–5.
The selected model settings are in `configs/selected.json`.

Training inputs use a 50% clean / 50% mixed-noise mixture, with noise strength
sampled from `U(0, 0.4)`. Validation inputs and prediction labels stay clean.
Each best checkpoint is evaluated on RW and Satellite at noise levels
`r = 0, 0.2, 0.4`.

RW and Satellite metrics are averaged equally within each seed, then summarized
across seeds. MAE/RMSE are stored in Ah and MAPE in percent.

```text
outputs/
├── results_per_domain.csv
├── results_per_seed.csv
├── results_macro.csv
└── jobs/<model>/seed_<seed>/
```

Add `--save-predictions` to save per-window forecasts. Use `--resume` to continue
an interrupted run. Use a new output directory when changing experiment settings.

## Parameter search

Search five candidate configurations per model using source-validation MAE:

```bash
python run.py --action tune --device cuda --output outputs/search --resume
python run.py --action train --config outputs/search/tuned_selected.json --device cuda --output outputs/retuned --resume
```

For a model subset, use the same `--methods` list in both commands.

## Structure

```text
├── run.py
├── baselines/
│   ├── models/
│   ├── factory.py
│   ├── protocol.py
│   └── workflow.py
├── configs/
├── data/
├── reference/
└── requirements.txt
```

## Acknowledgements

See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for the referenced model repositories.
