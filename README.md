# MSTGCNet

Multiscale Spatio-Temporal Graph Convolutional Network for UAV Anomaly Detection

**Note**: The complete project code will be released after the paper is accepted.

## Prepare ALFA 10-variable data

The selected ALFA files in `alfa_10vars/` can be converted to the format expected by
`Dataset_ALFA`. By default, files whose names contain `no_ground_truth` are excluded.
With this stricter source filter, the test split still keeps the paper-sized 24556
points and 0.25 anomaly rate, but the available normal train/validation rows are
slightly fewer than Table III.

```bash
python scripts/preprocess_alfa.py
```

This creates:

- `dataset/ALFA10vars/train.csv`: all available normal points outside the test
  window. `Dataset_ALFA` splits this internally into 90% training and 10%
  validation points.
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

The default configuration follows Table IV: `seq_len=96`, `winsize=96`,
`batch_size=128`, `learning_rate=1e-4`, `train_epochs=10`, `patience=3`,
`dropout=0.1`, `d_model=64`, three MSTGCNet blocks, four experts per block,
top-3 expert selection, top-3 Fourier bases, trend kernels `4,8,12`, patch
sizes selected from the paper's patch pool `2,6,8,12,16,32`, reconstruction
loss plus expert balance loss, and ATSSD adaptive thresholding.
The evaluation prints both raw point-wise metrics and point-adjusted metrics.

To use a fixed anomaly-ratio percentile threshold for ablation/debugging:

```bash
python run.py --threshold_method percentile
```

For a stricter flight-level split instead, use:

```bash
python scripts/preprocess_alfa.py --split-policy flight
```

To reproduce the Table III train/validation row counts exactly, include the
`no_ground_truth` file explicitly:

```bash
python scripts/preprocess_alfa.py --include-no-ground-truth
```
