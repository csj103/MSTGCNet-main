# MSTGCNet

Multiscale Spatio-Temporal Graph Convolutional Network for UAV Anomaly Detection

**Note**: The complete project code will be released after the paper is accepted.

## Prepare ALFA 10-variable data

The selected ALFA files in `alfa_10vars/` can be converted to the format expected by
`Dataset_ALFA`. By default, the preprocessing script matches the ALFA experimental
setting reported in the paper: 10 features, 52488 training points, 5833 validation
points, 24556 test points, and a 0.25 test anomaly rate.

```bash
python scripts/preprocess_alfa.py
```

This creates:

- `dataset/ALFA10vars/train.csv`: 58321 normal points. `Dataset_ALFA` splits this
  internally into 52488 training points and 5833 validation points.
- `dataset/ALFA10vars/test.csv`: 24556 continuous test points with 6139 anomalies.
- `dataset/ALFA10vars/train_meta.csv` and `test_meta.csv`: flight and segment
  metadata used to prevent sliding windows from crossing discontinuous flights.
- `dataset/ALFA10vars/split_summary.csv`: per-flight split and label summary.
- `dataset/ALFA10vars/metadata.json`: feature list and dataset statistics.

Use `--root_path ./dataset/ALFA10vars/` and set `--enc_in 10 --dec_in 10 --c_out 10`
when running the model on this 10-variable ALFA subset.

## Run ALFA Experiment

`run.py` now defaults to the ALFA 10-variable anomaly detection setting:

```bash
python run.py
```

The default configuration uses `seq_len=100`, `enc_in=10`, three MSTGCNet blocks,
DFT-based seasonal routing, multi-kernel trend routing, patch-level CC-STGCN
experts, reconstruction loss plus expert balance loss, and the paper's ATSSD
adaptive threshold.
The evaluation prints both raw point-wise metrics and point-adjusted metrics.

To use a fixed anomaly-ratio percentile threshold for ablation/debugging:

```bash
python run.py --threshold_method percentile
```

For a stricter flight-level split instead, use:

```bash
python scripts/preprocess_alfa.py --split-policy flight
```
