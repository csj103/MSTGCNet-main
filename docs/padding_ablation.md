# 输入卷积 Padding 单因素对照

只改变输入 Conv1d 的 `padding_mode`。两侧均填充 1 点，卷积核长 3；不改
Patch 的补齐、趋势池化、路由器、图学习、RevIN、损失、评分或 ATSSD。

| 实验 | experiment_mode | input_padding | 模型 | 训练 |
|---|---|---|---|---|
| E0 | E0 | default 或 circular | 原始 MSTGCNet | 使用已有结果 |
| E0-P | E0 | replicate | 原始 MSTGCNet | 从头训练 |
| E1 | E1 | default 或 replicate | 变量级模型 | 使用已有结果 |
| E1-P | E1 | circular | 变量级模型 | 从头训练 |

`E0-P` / `E1-P` 是对照实验名称，不是 `--experiment_mode` 的新取值。
原命令不传 `--input_padding` 时行为和目录名均保持原样。显式指定原生
Padding 也使用原目录名；切换 Padding 后分别使用 `_E0P_cfg...`、
`_E1P_cfg...` 标识，新配置哈希包含实际 Padding，不会使用原 E0/E1 的目录。
配置文件记录请求值，启动日志输出实际生效值。

## Linux 服务器执行

先把本次修改同步到服务器的 `exp/variable-granger-v1` 分支，在项目根目录执行：

```bash
python -m pytest tests -q --import-mode=importlib
bash scripts/run_padding_ablation.sh
```

脚本依次从头训练 E0-P、E1-P，每组训练完成后自动加载**本组**最佳验证集
checkpoint 并测试。训练前不加载任何已有权重，没有续训或微调。
日志为 `logs_padding/E0-P.log` 和 `logs_padding/E1-P.log`。
可用 `tail -f logs_padding/E0-P.log` 查看进度。终端容易断开时：

```bash
nohup bash scripts/run_padding_ablation.sh > padding-launch.log 2>&1 &
```

脚本复用现有 `dataset/ALFA10vars_20hz/`，不重新划分或预处理数据。
命令与本地收到的服务器 E0/E1 `experiment_config.json` 已核对，除了
Padding 和对应输出标识外相同；本地验证时 GPU 设备自动变为 CPU。
路径字符串也沿用服务器原命令，避免无意义的哈希变化。
`--granger_prior` 路径保留是为了与旧配置一致，E0/E1 不加载该先验。

共有配置：seed=2025、窗口 96、batch=128、最多 10 epoch、patience=3、
Adam lr=0.0001、dropout=0.1、3 层 × 4 专家、原 Patch 列表、top_k=3、
d_model=64、d_var=16、causal_last、ATSSD。保留旧 E0/E1 结果，不需要重跑。
请使用同一批数据文件与服务器环境。再次执行同组命令会从头训练并覆盖
该 Padding 对照自己的目录；需要多随机种子时明确修改 `--seed`。

## 单独运行（也适用于 Windows CMD）

在已激活训练环境的项目根目录，每行分别是一条完整命令：

```text
python -u run.py --is_training 1 --model MSTGCNet --experiment_mode E0 --input_padding replicate --root_path ./dataset/ALFA10vars_20hz/ --granger_prior ./analysis/variable_granger_v1/prior/granger_prior.npz --d_var 16 --lambda_prior 1 --seed 2025 --score_mode causal_last --threshold_method atssd --use_gpu true --gpu 0 --num_workers 0 --report_diagnostics true
python -u run.py --is_training 1 --model MSTGCNet --experiment_mode E1 --input_padding circular --root_path ./dataset/ALFA10vars_20hz/ --granger_prior ./analysis/variable_granger_v1/prior/granger_prior.npz --d_var 16 --lambda_prior 1 --seed 2025 --score_mode causal_last --threshold_method atssd --use_gpu true --gpu 0 --num_workers 0 --report_diagnostics true
```

模型与路由记录位于 `checkpoints/<setting>/`；指标、逐点分数、阈值和事件
记录位于 `label_results/<setting>/causal_last/`。`<setting>` 分别带有
`_E0P_cfg` / `_E1P_cfg`，完整名称会打印在训练日志中。

## 比较和验证边界

先比较 E0→E0-P、E1→E1-P，判断 Padding 在各自结构中的作用；再比较
E0-P 与 E1（同为 replicate）、E0 与 E1-P（同为 circular）。后两对仍有
表示结构、容量、路由接口等差异，不能据此单独归因到某个模块。

主要比较 PR-AUC（当前实现为 average precision）、Raw F1，同时检查
Raw Precision/Recall、ROC-AUC、事件命中/延迟和正常验证集分数。
不能用降低正常末点重构误差直接推断异常检测更好。

CPU 自动测试覆盖：初始 state_dict 逐项一致、参数量一致、模块唯一差异、
卷积内部时间点完全一致而首末点变化、旧目录名兼容、新目录隔离、输出
形状、有效梯度、三层更新传递、固定权重推理复现。本地不进行完整训练，
也不把此前冻结权重切换 Padding 的探针结果当作本次重训结果。

2026-10-28 本地验证：完整测试集 **65 passed**（其中新增 Padding 测试 8 项）；
E0 与原始基线、E1 与修改前版本的默认权重/输出逐项一致，旧服务器目录名
均可复现。两组脚本参数及训练/验证数据 SHA256 与服务器记录一致。
详细记录见 `analysis/padding_ablation_v1/verification.json`。未进行 GPU 完整训练。
