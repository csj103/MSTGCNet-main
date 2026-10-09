# MSTGCNet

## V12 CC-STGCN 消融：去掉图传播

在本项目目录运行，使用同一份已经生成的 ALFA 数据：

```powershell
# 原版 V12 基线（训练并评估）
python run.py

# 消融 A：训练并评估不含图传播的模型
python run.py --without_graph

# 加载消融模型自己的最佳权重进行评估
python run.py --without_graph --is_training 0
```

`--without_graph` 等价于 `--abl_GCN 1`，沿用原有消融分支。
保留变量投影、所有 patch 尺度、块内时间注意力、逆 patch、LayerNorm、
FFN、GMoE 路由及残差；仅跳过邻接矩阵构建、邻居聚合、图线性变换和
图分支残差。它不是移除整个专家，也不是仅去掉因果掩码。

实验名称自动增加 `_no_graph`，权重、图表和预测结果与默认基线分开。
旧的 `--abl_GCN 1` 现在也使用该标记，已有旧消融目录不会被自动迁移。
默认基线的名称和配置指纹不变。应从头训练消融模型，不能将完整模型
权重直接关闭图后评估当作本次训练消融。重复同一命令仍会使用同名目录；
若需保留多次运行，请指定不同的 `--implementation_tag`。

两组保持原版 V12 的种子 2025、数据划分、归一化、训练参数、重构与均衡
损失、验证准则、末点评分和 ATSSD 不变。可以保留 `paper_strict=true`：
该检查只约束共同的超参数，并不表示消融结构是完整论文模型。

为保持初始化随机数顺序和 state_dict 兼容性，图参数仍然注册，但消融
前向不会使用它们，也不会为它们产生梯度。因此总参数量不会下降，不能
据此声称参数压缩。比较优先查看 raw 指标，point-adjusted 指标依赖真值。
原版硬 Top-k 图嵌入无梯度、路由负载概率偏移问题保持原样，因此结论
应表述为“原版 V12 图传播分支的贡献”，而非作者完整实现的严格消融。

Multiscale Spatio-Temporal Graph Convolutional Network for UAV Anomaly Detection

**Note**: The complete project code will be released after the paper is accepted.

## Prepare ALFA 10-variable data

The selected ALFA files in `alfa_10vars/` can be converted to the format expected by
`Dataset_ALFA`. By default, files whose names contain `no_ground_truth` are excluded.
The default `fine_grained` policy assigns complete flights to one split only and
holds out at least one labelled flight for every fine-grained anomaly type
available in the local ALFA subset. Training and validation contain normal rows
only, so the threshold and early stopping do not see fault labels.

```bash
python scripts/preprocess_alfa.py
```

This creates:

- `dataset/ALFA10vars/train.csv` and `val.csv`: normal points from disjoint
  training and validation flights.
- `dataset/ALFA10vars/test.csv`: complete held-out fault flights covering all
  fine-grained ALFA anomaly types found in `alfa_10vars/`.
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

The paper-aligned model and training defaults are checked at startup by
`--paper_strict true`.
Changing a Table IV setting raises an error; pass `--paper_strict false` only
for ablations. The new implementation uses the `v12_finegrained` experiment tag so
that checkpoints produced by earlier incompatible model definitions cannot be
loaded accidentally.

For deployable pointwise detection, use the causal normal-history threshold
and calibrate each variable by its normal training reconstruction error:

```bash
python run.py --threshold_method causal_atssd --score_normalization train_feature --paper_strict false --implementation_tag point_v12_finegrained
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

For the older balanced-by-coarse-fault diagnostic split, use:

```bash
python scripts/preprocess_alfa.py --split-policy fault_balanced
```

To reproduce the Table III train/validation row counts exactly, include the
`no_ground_truth` file explicitly:

```bash
python scripts/preprocess_alfa.py --split-policy paper --include-no-ground-truth
```
