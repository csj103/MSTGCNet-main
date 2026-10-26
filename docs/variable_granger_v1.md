# MSTGCNet Variable–Granger V1

## 1. 基线与交付状态

工作目录：`D:\研究生文件\代码复现\MSTGCNet-main-MSTGCNet-v12-`。

基线：远程 `MSTGCNet-v12原始复现` 分支，提交
`5382d528ff93d03f5a30e3b34563c4e70803f5e0`。此目录最初是无 `.git`
的 GitHub 源码副本；建立 Git 后以 mixed reset 关联该提交，没有改写源码文件。
初始状态与远程提交完全一致。开发分支为 `exp/variable-granger-v1`。
原始分支未修改，未推送远程，未修改 DTSGAD。

第一版已实现 E0–E5，执行 CPU smoke、梯度检查、真实正常训练窗口的小规模验证。
**尚未做完整训练、GPU 验证或完整真实测试集检测评估，因此没有性能提升结论。**
训练入口的评分与阈值协议由原工程保留，已用独立合成数据跑通训练/保存/加载/
评分/报告的集成测试。集成测试分数不是 ALFA 实验结果。

## 2. 数据流与真实变量身份

原版 E0：

```text
[B,96,10] -> 逐变量 RevIN
 -> 跨变量 Conv1d(10,64) -> [B,96,64]
 -> 3 个原始 GMoE
    专家 to_vars(64,10) -> [B,N,p] -> 时间注意力
    -> 原固定嵌入二值 Top-K 图 -> 图传播 -> to_model(10,64)
 -> Linear(64,10) -> RevIN 逆变换 -> [B,96,10]
```

E1–E5：

```text
[B,96,10] -> 逐变量 RevIN
 -> 将变量作为独立样本 [B*10,1,96]
 -> 共享时间 Conv1d(1,d_var,k=3) + 固定索引的可学习变量身份嵌入
 -> [B,96,10,d_var] + 时间位置编码
 -> 3 个 GMoE，层输入和输出始终为 [B,96,10,d_var]
    路由：逐变量 DFT/趋势/隐藏投影，再均值聚合窗口路由特征
          保留原噪声门控、Top-K=3、负载与专家均衡损失
    专家：沿时间划分 -> [B,N,p,d_var]
          共享 Patch 内时间注意力，添加输入残差，屏蔽补零位置
          -> 对每个真实变量 Patch 节点执行图传播
          -> Patch 还原、逐变量特征 LayerNorm/FFN
          -> [B,96,10,d_var]
    稀疏路由选择对应样本，混合专家输出并保留 GMoE 残差
 -> 每变量共享 Linear(d_var,1) -> [B,96,10]
 -> RevIN 逆变换
```

默认 `d_var=16`，支持 32。没有 `10→64→10` 的物理变量重建式映射。
编码、时间注意力、FFN、重构头仅作用于时间或最后的特征维，不收缩/重排变量轴。
只有显式图传播可以汇聚不同变量。下一层使用上一层实际更新的变量特征；
测试验证了相邻层输入与上层输出逐位相同，并非重复使用原始窗口。

所有专家保持基线 Patch 配置：

| 层 | Patch size | 节点数 N=ceil(96/p)×10 |
|---|---|---|
| 1 | 8,12,16,32 | 120,80,60,30 |
| 2 | 6,8,12,16 | 160,120,80,60 |
| 3 | 2,6,8,12 | 480,160,120,80 |

节点索引为 `patch_id*10+variable_id`。测试还覆盖 p=7 和 p=128 的补零/截断。

## 3. 数据划分与采样

先执行原始预处理的 `fine_grained`、seed=2025，保留原有文件级划分。
原始行数为 train=54,440、val=9,923、test=14,279。
正常训练片段采样间隔的中位数范围为 0.047546–0.051623 秒，但间隔变化显著，
不能把 `freq=s` 视为 1 Hz，也不能直接认为原数据已规则采样。
本次检查的有标签源文件 10 个特征列没有缺失值。

`scripts/regularize_alfa.py` 新建独立的 20 Hz 数据版本：

- 按原连续 segment、source_file、原始索引断点及时间间隔分别处理；
  时间非递增或大于 0.25 秒的间隔新开片段，绝不跨片段填充。
- 规则时间网格上使用最近过去观测值保持（ZOH），不做线性插值，
  因此 `wp_dist` 等阶跃字段不会生成中间值。
- 标签也使用最近过去观测标签。事件起点可能量化到下一网格点，误差小于 0.05 秒；
  极短于一个网格的事件可能无法保留，正式换采样率时需重查事件数。
- `*_meta.csv` 保留 `original_index`、`original_time_sec`、`input_split_row`，
  `original_sampling_audit.csv` 保存原间隔统计。
- 本次规则版 train=56,373、val=10,331、test=14,627；test 异常点 2,617，7 个事件。

**E0–E5 主消融必须使用相同的 20 Hz 版本。** E0 同时可在原始行网格独立运行以核对
旧基线，但该结果与 20 Hz 版本不可直接归因于模型结构变化。规则版计数与论文/历史
结果不同，文档与元数据明确记录，没有将其当作精确论文数据复现。

## 4. 正常 Granger 先验

实现 `linear_var`，未加入非线性预测梯度方法，以维持首版范围。

`scripts/build_granger_prior.py` 只读取 `train.csv`、`train_meta.csv` 与数据集元数据。
`val.csv` 只检查是否存在以匹配 loader 的分割逻辑，不读取其数据；
没有读取测试数据或测试标签的代码路径。若没有独立验证集，按 loader 只取前 90%。
默认进一步限制为训练集中 `fault_type=normal` 的飞行；
可显式 `--include-prefault` 纳入训练集的正常故障前片段，但本次未启用。

默认配置：

- 20 Hz，最大滞后 5 个采样点（0.25 秒）。
- 每个连续正常片段独立取 512 点、不重叠窗口，绝不拼接飞行。
- 默认一阶差分，减少水平序列的趋势影响；可选 `--transform level`，必须单独标记。
- 用全部变量的 1…5 阶历史预测目标。逐个移除来源变量的全部滞后项，
  比较受限/非受限 OLS 的 SSR，依据实际矩阵秩计算嵌套 F 检验自由度。
  常量/冗余变量安全跳过；精确预测使用 F 统计量的数值极限，不丢弃强边。
- 每窗口对 90 条非对角方向边做 BH-FDR=0.05，要求相对 SSR 改善至少 0.01。
- 先在飞行内统计，再对飞行等权平均；支持率至少 0.30 且至少 2 个飞行出现。
  权重为显著窗口 SSR 改善乘出现指示的飞行内/跨飞行均值。
- `G[j,i]` 表示 `i→j`，方向独立估计，不做对称化，不使用相关/Coherence 替代。

检验采用受限与非受限回归比较的 Granger 思路，可参见
[statsmodels 的官方说明](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html)。
本实现为条件于全部变量历史的多变量版本。线性预测依赖不等于物理干预因果；
差分/FDR 也不能完全消除非平稳、隐藏混杂及 ZOH 带来的统计影响。

本次从 8 个正常训练飞行的 14 个窗口得到 7 条稳定边：

| 来源 | 目标 | 权重 | 飞行等权窗口支持率 |
|---|---|---:|---:|
| nav_z | alt_error | 0.059641 | 0.8750 |
| alt_error | aspd_error | 0.056031 | 0.7083 |
| xtrack_error | wp_dist | 0.067111 | 0.3750 |
| nav_z | nav_pitch | 0.133664 | 1.0000 |
| nav_x | nav_yaw | 0.043788 | 0.3750 |
| nav_y | nav_yaw | 0.034619 | 0.3958 |
| nav_yaw | nav_y | 0.018877 | 0.3542 |

这些是本次计算结果，样本量仍有限，不能直接宣称具有物理因果真实性。
产物在 `analysis/variable_granger_v1/prior/`：NPZ、metadata、matrix/stability CSV、
逐窗口边、逐飞行稳定性、稳定边 CSV、采样审计和热力图。
保存变量顺序、方向、滞后、采样、变换、源文件与训练文件 SHA256；
正式入口拒绝合成测试先验，核对数据/元数据哈希、变量顺序及采样信息。

## 5. 先验映射、动态图和消融

Patch 节点先经时间注意力得到 `[B,N,p,d_var]`，展开每节点的 `p*d_var`
特征后投影至默认 16 维 Query/Key：

```text
S_dynamic[b,target,source] = Q(H_target) K(H_source)^T / sqrt(graph_dim)
S_fuse = S_dynamic + lambda_prior * P
```

扩展先验 `P[target_node,source_node]` 使用真实变量索引查 `G[target_var,source_var]`。
仅当来源 Patch 严格早于目标 Patch，并且两 Patch 的可能时间差范围与 `[1,max_lag]`
相交时添加先验。对 Patch 距离 delta，此下界为 `(delta-1)*p+1`。
这只是粗尺度滞后支持映射，不宣称整块对齐于精确 VAR 延迟。
同 Patch 的先验项为零；同 Patch 动态连接属于统计关联。

所有图模式均禁止 `source_patch>target_patch`。动态图排除自身后选择 K 个邻居，
将选中连续 logits scatter 回矩阵，添加固定 self logit=0 后逐行 softmax；
未选中边本步梯度为零，但选中边的 Query/Key 有有效梯度。
非先验边仍可选，先验不会成为永久禁止连接的掩码。
传播用 gather 稀疏聚合 K+1 条边，保留节点原信息、专家残差和 GMoE 残差。
没有结构异常分数、因果损失或 PATK。

| 模式 | 表示 | 图机制 |
|---|---|---|
| E0 | 原始 64 通道 | 原始图与全部原始模型路径 |
| E1 | 逐变量 | 原固定节点嵌入、二值 Top-K、原度归一化 |
| E2 | 逐变量 | 当前输入的 Q/K 动态图 |
| E3 | 逐变量 | 固定历史 Granger 图，正权边 Top-K+self，逐行归一化 |
| E4 | 逐变量 | Q/K + lambda_prior×Granger |
| E5 | 逐变量 | E4，但对先验行/列使用同一个固定随机排列 |

`lambda_prior=0` 的 E4 在对齐相同模型权重后与 E2 完全相同；
测试还修改全部 expanded_prior 后确认输出不受影响。
E5 使用单独生成器、seed=2025，排列保存为 checkpoint buffer，不扰动权重初始化 RNG。
E3 固定图不需要 Q/K 梯度；E1 为保留原图算法，其固定节点嵌入经过二值选择无关系梯度，
这也是 E0 原实现已有特性。E2/E4/E5 的 Q/K 梯度已实测非零。

**因果保证范围**：专家内部图传播屏蔽未来 Patch。共享时间卷积使用相邻时间步，
原 RevIN 和路由使用完整输入窗口，因此整模型的历史隐藏状态不是逐时刻因果的。
保留的 `causal_last` 仅用截止当前点的窗口计算最终点分数，没有读取窗口外未来观测。
不要把这套重构模型描述成对所有历史输出都端到端因果。

## 6. 已执行验证与成本

完整测试：`python -m pytest tests -q --import-mode=importlib`。
本次最终执行结果为 **57 passed**。
详细结果见 `analysis/variable_granger_v1/verification/verification.json`。

- E0 相对于基线提交：state_dict 键/值、固定种子初始化、输出、辅助损失、默认 setting
  逐位一致。训练损失仍是每窗口平方误差求和后取 batch 均值，均衡损失只加一次。
- E0–E5 的 `[2,96,10]` 前后向均有限，输出维度一致；eval 重复推理逐位一致。
- 逐变量编码隔离、所有层特征连续传递、Patch 还原/节点计数/补零、方向/未来屏蔽、
  固定图、当前输入依赖、Query/Key 梯度、零先验等价、E5 排列、检查点回读均有测试。
- 合成有向 VAR 正反方向、常量列、精确滞后预测、训练文件读取隔离、ZOH/片段隔离、
  原事件起点延迟计算与完整训练/评分流程均有测试。
- E1–E4：相同 seed、32 个固定分布于真实正常训练集的窗口，8 步 Adam、batch=4；
  32 个验证窗口，训练/验证均有限。验证重构约为
  E1=1077.3923、E2=748.1133、E3=842.2547、E4=748.4753。
  这些只证明训练链可运行，不能用于检测性能排序。

参数总数：E0=463,144，E1–E5 各 336,938。
为保持结构/初始化可比，E1–E5 分配相同模块；其中部分模块在某模式不激活。
按可微路径统计：E0=451,624，E1/E3=254,762，E2/E4/E5=325,418。
报告保存具体 dormant 参数名。模式差异仍然具有有效容量差别，不能只比较总参数。

CPU 单窗口 eval 的最近实测结果保存在 verification.json；2 线程、batch=1、
warmup=2、重复=5，均为几十毫秒量级。此微基准会随负载/路由改变，
不能把几毫秒差异解释成算法优劣。GPU 推理和吞吐未验证。

E4 CPU batch=16 前后向实测约 1.34 秒、进程峰值工作集约 896 MiB，
见 `memory_batch16.json`。邻居传播稀疏，但 Q/K 分数仍为稠密 `B×N×N`：
N 最大 480，batch=128 的单个 float32 分数张量约 112.5 MiB；
训练还保存中间梯度，不能据 batch=16 推断默认 batch=128 的 GPU 内存一定足够。
默认训练 batch=128/10 epochs、3 层、4 专家、原 Patch、top_k=3、学习率等不变。

## 7. Windows CMD 命令

以下命令在 CMD 内执行；新目录已提供生成的数据/先验，不必覆盖重做。
原数据预处理沿用原程序；规则化脚本为防误覆盖会拒绝已存在的输出目录。

```bat
cd /d D:\研究生文件\代码复现\MSTGCNet-main-MSTGCNet-v12-
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
python scripts\preprocess_alfa.py --source alfa_10vars --output dataset\ALFA10vars --split-policy fine_grained --seed 2025
python scripts\regularize_alfa.py --source dataset\ALFA10vars --output dataset\ALFA10vars_20hz --hz 20
python scripts\build_granger_prior.py --root dataset\ALFA10vars_20hz --output analysis\variable_granger_v1\prior --method linear_var --max-lag 5 --window 512 --stride 512 --transform difference --stability 0.3 --min-flights 2
python -m pytest tests -q --import-mode=importlib
python scripts\check_variable_granger.py --steps 8 --batch-size 4 --threads 2
```

GPU 全训练，单个 E4（训练后自动测试）：

```bat
scripts\run_variable_granger.cmd E4 true
```

严格对照 E0–E5 都从头训练，依次执行：

```bat
scripts\run_variable_granger.cmd ALL true
```

也可以逐个选择 `E0`、`E1`、`E2`、`E3`、`E4`、`E5`。启动器统一数据版本、
seed=2025、`causal_last`、ATSSD、d_var=16、lambda_prior=1，并打开报告。
需要 CPU 全训练可将第二个参数设为 false，但当前任务未运行这种全训练。

直接命令（将 E4 换为 E0–E5；E3/E4/E5 必须给真实先验）：

```bat
python run.py --model MSTGCNet --experiment_mode E4 --root_path .\dataset\ALFA10vars_20hz\ --granger_prior .\analysis\variable_granger_v1\prior\granger_prior.npz --d_var 16 --lambda_prior 1 --seed 2025 --score_mode causal_last --threshold_method atssd --use_gpu true --num_workers 0 --report_diagnostics true
```

仅测试对应完整训练的检查点：保持训练时其余参数与路径文本一致，添加 `--is_training 0`。
`report_diagnostics` 不进入指纹，可在测试时打开。模型、图超参数、数据路径、seed、先验
文件 SHA256 等进入设置指纹。更换先验或训练参数将生成不同检查点目录。

原始数据网格 E0 单独复现：

```bat
python run.py --experiment_mode E0 --root_path .\dataset\ALFA10vars\ --use_gpu true --num_workers 0
```

可选 d_var=32 或 lambda_prior=0 只修改对应参数并从头训练。
如果因 GPU 内存减小 batch，需 `--paper_strict false`，并且所有对照使用同样 batch；
必须报告这个训练协议变化，不与默认 batch 实验混称。

可选完整真实测试集评分的 CPU 小训练检查为
`python scripts\check_variable_granger.py --evaluate`。本次未执行这个选项。
它只训练 8 步，输出前缀含 `cpu_smoke`，不能当作正式实验结果。

## 8. 实验产物与未验证项

正式 `run.py` 结果：

- `checkpoints/<setting>/experiment_config.json`、`checkpoint.pth`、
  `routing_epoch_<epoch>.json`：完整参数、权重、每层每专家次数/比例/每窗口选择率。
  配置另保存 Git 分支/提交/工作区状态、训练入口、源码和训练/验证文件 SHA256。
- `label_results/<setting>/causal_last/metrics.json`：ROC-AUC、PR-AUC（Average Precision）、
  Raw Precision/Recall/F1、补充 PA-F1、事件命中、参数量及可选 CPU 计时。
- 同目录原始 `test_energy/test_labels/test_indices/raw_pred/adjusted_pred/threshold.npy`。
- `raw_metrics_by_anomaly_type.csv`：故障类型的 Raw P/R/F1、ROC-AUC 和 AP；
  单一类别时 AUC 为空。`adjusted_metrics_by_anomaly_type.csv` 为 GT-assisted 补充。
- `event_details.csv`：按完整 test 元数据定位事件真实起点、覆盖点数、首命中、
  延迟点数/秒；未命中为空，不记为零延迟。每段最初 95 点通常不评分，覆盖单列记录。
- `graphs/`：已训练模型在代表窗口的邻接 NPY/PNG 和 manifest，
  记录专家实际路由选中样本索引、节点顺序和方向。变量汇总图是传播权重，不是因果图。

本次 CPU 小验证产物在 `analysis/variable_granger_v1/verification/`：
总报告、内存报告、E1–E4 的 small_training.json、代表动态图。
本地 `smoke_checkpoint.pth` 仅用于调试，已忽略 Git，不应当作 GPU 完整训练检查点。
生成数据集因原仓库 `.gitignore` 不纳入 Git；复制到 GPU 时可携带整个 dataset/ 和 prior/
或严格按命令重建。正式入口会发现训练数据/先验错配并拒绝运行。

未验证：GPU/CUDA、多 GPU、混合精度、batch=128 内存、完整训练收敛、
真实测试集 Raw F1/PR-AUC 改善、跨 seed 稳健性、Granger 物理合理性和非线性 Granger。
数据划分、原重构损失、专家均衡损失、异常分数与 ATSSD 实现保持原逻辑；
仅新增诊断输出及修正测试时自定义 checkpoints 目录的读取。

## 9. 文件清单

修改：`models/MSTGCNet.py`（提取路由接口、广播适配、Model 工厂、图样本索引）；
`run.py`（模式/参数/来源核验/指纹/seed）；`data_provider/data_loader.py`
（实验模式的 CSV 变量顺序核验）；`exp/exp_anomaly_detection.py`
（路由落盘、诊断报告、自定义检查点读取）；`.gitignore`（本地大调试产物）。

新增：`models/variable_granger.py`、`layers/GrangerPrior.py`、
`scripts/build_granger_prior.py`、`scripts/regularize_alfa.py`、
`scripts/check_variable_granger.py`、`scripts/run_variable_granger.cmd`、
`utils/experiment_reporting.py`、`tests/test_variable_granger.py`、
`tests/test_granger_prior.py`、本说明、实施计划及 `analysis/variable_granger_v1/` 报告。
