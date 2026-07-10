# MSTGCNet

Multiscale Spatio-Temporal Graph Convolutional Network for UAV Anomaly Detection

**Note**: The complete project code will be released after the paper is accepted.

## Prepare ALFA 10-variable data

The selected ALFA files in `alfa_10vars/` can be converted to the format expected by
`Dataset_ALFA`. By default, files whose names contain `no_ground_truth` are excluded.
The default `fault_balanced` policy assigns complete flights to one split only.
It selects whole fault flights whose anomaly totals are closest to the scarcest
fault type, then splits the remaining flights into normal-only training and
validation sets. This avoids flight leakage and prevents engine/aileron faults
from dominating the test metrics.

```bash
python scripts/preprocess_alfa.py
```

This creates:

- `dataset/ALFA10vars/train.csv` and `val.csv`: normal points from disjoint
  training and validation flights.
- `dataset/ALFA10vars/test.csv`: complete held-out fault flights with balanced
  anomaly totals across engine, elevator, aileron, and rudder faults.
- `dataset/ALFA10vars/*_meta.csv`: flight and segment
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
top-3 expert selection, top-3 Fourier bases, trend kernels `4,8,12`, and the
coarse-to-fine patch assignment `[8,12,16,32]`, `[6,8,12,16]`,
`[2,6,8,12]`. The first block is confirmed by Fig. 7; the remaining two
blocks are an explicit reproduction assumption based on the reported patch
pool. Training uses reconstruction loss plus expert balance loss.
The evaluation prints both raw point-wise metrics and point-adjusted metrics.
It also reports ROC-AUC, PR-AUC, and the number of anomaly events hit by the
raw predictions. Point-adjusted metrics use ground-truth anomaly boundaries
and are therefore not a deployable streaming metric.

The default `causal_last` mode moves the 96-point model window by one timestamp
and uses only the reconstruction error at the final timestamp. ATSSD then uses
the latest 96 causal scores and resets at every flight-segment boundary. The
first 95 points of each segment do not have a complete model window and are
excluded; `test_indices.npy` maps every score back to its original test row.

For the non-overlapping, flattened scoring path visible in the authors'
released experiment scaffold, use:

```bash
python run.py --score_mode paper_nonoverlap
```

The paper-aligned defaults are checked at startup by `--paper_strict true`.
Changing a Table IV setting raises an error; pass `--paper_strict false` only
for ablations. The new implementation uses the `v10_balanced` experiment tag so
that checkpoints produced by earlier incompatible model definitions cannot be
loaded accidentally.

For deployable pointwise detection, use the causal normal-history threshold
and calibrate each variable by its normal training reconstruction error:

```bash
python run.py --threshold_method causal_atssd --score_normalization train_feature --paper_strict false --implementation_tag point_v10_balanced
```

Unlike paper ATSSD, `causal_atssd` feeds a clipped form of detected high-score
points into its 96-point reference window. This preserves persistent high-score
fault alarms while allowing gradual normal regime shifts to update the baseline.
For a precision-oriented experiment, lower
`--alpha` and require consecutive candidates, for example
`--alpha 0.001 --alarm_confirmation 3`. The optional `--latch_alarm 1` is only
appropriate when faults are known to persist until the end of a flight segment.
Experiment directories use a compact configuration fingerprint to stay within
Windows path limits. The complete resolved arguments are saved as
`checkpoints/<setting>/experiment_config.json`.

See `PAPER_ALIGNMENT.md` for the equation-by-equation alignment status and the
implementation details that the paper and placeholder repository do not expose.

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
python scripts/preprocess_alfa.py --split-policy paper --include-no-ground-truth
```
