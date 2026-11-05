# 日前电价预测

首版已完成。以下实验报告与运行命令覆盖数据版本、滚动样本外预测及误差评估。命令均从仓库根目录运行。

## 4. 模型研发路线

采用 Python 与 PyTorch 作为主要技术栈。所有模型使用同一数据版本、预测时点和评估区间比较。

| 阶段 | 模型 | 目的 |
| --- | --- | --- |
| 基线 | 前一日/前一周同一时段价格、Ridge 或 LEAR、LightGBM | 建立季节性与强监督学习参照 |
| 首个深度模型 | 多变量 LSTM，编码历史窗口并融合未来已知特征 | 验证多源序列建模和直接多步输出 |
| 注意力模型 | Transformer；进一步比较 TFT 或 PatchTST | 研究较长历史依赖及外生信息融合 |
| 扩展实验 | TCN、分位数模型或模型融合 | 按基准结果决定是否推进 |

首期点预测优先比较 MAE/Huber 损失，后续使用 Pinball Loss 训练分位数。模型复杂度由验证结果和训练成本决定，不预设 Transformer 必然优于简单模型。

消融实验依次比较：仅历史价格 → 加日历 → 加负荷/新能源预测 → 加当时可得的天气预报（待接入），并比较历史窗口和预测步长。天气或风光字段尚不可用时，不宣称已完成对应的多源实验。

### 4.1 首轮基线运行

训练依赖与标准库下载脚本分开安装，建议 Python 3.9–3.12。以下命令在本仓库根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/run_baselines.py
```

macOS 的 LightGBM 需要 OpenMP 运行库。标准安装方式为 `brew install libomp`；本次机器已在 `/opt/miniconda3/lib` 提供该库，因此实际使用以下命令，无需修改系统安装：

```bash
DYLD_LIBRARY_PATH=/opt/miniconda3/lib .venv/bin/python scripts/run_baselines.py
```

运行参数在 [configs/baselines.json](../../configs/baselines.json)，实现位于 [scripts/run_baselines.py](../../scripts/run_baselines.py)。默认输出目录 `outputs/baselines_v1/`；重跑时指定新的空目录，例如 `--output outputs/baselines_repeat`，以保留此前结果。`requirements.txt` 固定直接依赖，`requirements-lock.txt` 记录本次 Python 3.9 环境的完整依赖版本。

首轮三种方法：

- **WeeklyNaive**：前一周同一交付日小时的真实日前价格。
- **LEAR**：价格与日历的 LASSO 自回归变体，每小时独立拟合，并在每次训练窗口内标准化。不是 epftoolbox 原样复刻，未加入其外生供需特征或非线性价格变换。
- **LightGBM**：跨小时共享的 L1 回归模型，增加小时编码和同小时滞后摘要，使用与 LEAR 相同的历史价格和日历信息。

统一输入是前 1/2/3/7 日的完整价格曲线、前 7 日逐小时均值与标准差、星期、年周期及源数据日历标记。首轮不使用供需字段和 ERA5 天气。预测时点约定为交付日前一天 12:00，并假设此前交付日的日前价格已发布；源数据无实际发布时间，该假设仍需核验。

初始训练采用 2024 年，验证期为 2025-01-01 至 2025-06-30，测试期为 2025-07-01 至 2025-12-30。因需要 7 天历史，最初 7 天不构成训练样本。每 7 天使用扩展窗口重训，训练标签严格早于预测交付日；每个交付日一次性预测 24 点。参数仅按验证期滚动 MAE 选择，进入测试前冻结，测试期间仅按既定计划更新模型参数。测试截止日避开尾部连续缺失与重复填充口径变化，未覆盖完整年度季节。

产物包含逐小时预测、整体及分组指标、参数选择、训练截止时间日志、模型权重、代码与配置快照、数据哈希及对比图。标签不补齐、不插值；所有模型在相同有效样本上评分，尖峰阈值固定为初始训练价格的 95% 分位数。

### 4.2 首轮实测结果

测试期 2025-07-01 至 2025-12-30，共 183 天、4,392 个小时点。MAE/RMSE 保留原始价格单位（具体单位仍待源方核验），数值越低越好；负价 F1 越高越好。

| 模型 | MAE | RMSE | rMAE | 相对周基线 MAE 改善 | 负价 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 前一周同小时 | 114.372 | 174.502 | 1.0000 | 0.00% | 0.4007 |
| LEAR | 83.621 | 124.204 | 0.7311 | 26.89% | 0.0067 |
| LightGBM | 81.777 | 127.062 | 0.7150 | 28.50% | 0.1932 |

LightGBM 总体 MAE 最低，LEAR 的 RMSE 更低；两个学习模型的负价识别都弱于周基线。结果仅反映当前价格+日历特征及本次时间区间，不能据总体误差改善推断交易收益。完整配置、分场景指标、限制及核验记录见 [首轮测评报告](../../reports/baselines_v1.md)，机器可读指标见 [指标 JSON](../../reports/baselines_v1_metrics.json)。本轮 14 项离线测试通过，全部逐小时预测和重训练截止时间已核对。

### 4.3 LSTM / TFT：CPU 训练与测评

实现位于 [deep_models.py](../../scripts/deep_models.py)，训练入口为 [run_deep_models.py](../../scripts/run_deep_models.py)，配置见 [deep_models.json](../../configs/deep_models.json)。在现有 `.venv` 安装更新后的依赖后运行：

```bash
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/run_deep_models.py

# 保留已有实验，重跑使用新的空目录
.venv/bin/python scripts/run_deep_models.py --output outputs/deep_models_repeat
```

全程强制 CPU，不自动切换 CUDA 或 macOS MPS。默认 2 个计算线程、隐藏维度 16、batch size 64、随机种子 42/43/44；AdamW + L1 损失、梯度裁剪与验证早停。每个样本使用过去 168 小时的价格和日历，预测下一交付日 24 点；未来输入仅含日历、前一日/前一周同小时价格与前 7 日均值/标准差，不使用目标日真实价格、供需字段或再分析天气。

LSTM 使用历史编码器和未来协变量解码器。TFT 按[原论文](https://arxiv.org/abs/1912.09363)实现变量选择网络、门控残差、LSTM 编解码、共享 V 的可解释多头因果注意力，以及逐时输出。此处只有一个市场，没有静态协变量，省略静态编码器；以 L1 学习中位数点预测，暂未输出分位数区间。

初始训练截止 2024-12-31，完整 2025 上半年验证集用于选择各模型各种子的最佳 epoch（上限 40，耐心 8）；这是固定训练截止的模型选择阶段，与基线滚动验证的口径不同。所有 epoch 选择在测试开始前保存并冻结。测试沿用 2025-07-01 至 2025-12-30，每 7 天使用当时全部历史有效样本从头训练，标准化也仅在本次训练数据拟合。三个种子分别报告指标，并额外报告等权预测集成。固定了结构和数据划分，测试结果不参与再调参。

默认读取现有 `outputs/baselines_v1/`，核对数据哈希和逐小时真实值后，与 LEAR / LightGBM 配对比较。因此先运行基线脚本或保留此前基线产物。预测时点和源标签定义沿用基线假设，仍需源方核验。

产物位于 `outputs/deep_models_v1/`：

- `training.jsonl` 实时追加每轮训练/验证 MAE、样本数、标签截止时间和耗时；完成后生成 `training.csv` 与 `training_curves.png`。
- `checkpoints/*.pt` 保存每次训练的 CPU 权重、标准化参数、模型配置、种子和特征定义；`selected_epochs.json` 记录进入测试前的选择。
- `validation_predictions.csv` / `test_predictions.csv` 包含逐小时真值、单种子预测与集成预测；`metrics.json` / `group_metrics.csv` 保存整体、小时、月份、季节、节假日和周末指标。
- `timing.json` 保存每次训练的轮数、耗时及标签截止时间；`metadata.json`、代码/配置/依赖快照保留复现信息。
- `tft_interpretability.npz` 保存每个种子最后一次重训模型在末个测试日的变量权重与时间注意力；这些权重不构成因果解释。

权重可通过 `torch.load(path, map_location="cpu", weights_only=True)` 读取，按配置调用 `scripts.deep_models.make_model` 并加载 `state_dict`。标准化参数保存在同一检查点的 `scaler` 字段，列表转换为 `np.float32` 数组后可交给 `transform` 和 `predict` 复用。这些是各回测截止日期的审计模型。

完整实测结果见 [CPU 深度模型报告](../../reports/deep_models_v1.md) 和 [机器可读指标](../../reports/deep_models_v1_metrics.json)。首轮仅比较一个小型结构配置和三个随机种子，尚未完成供需、天气或窗口长度的消融实验。

实测全流程 CPU 耗时约 **11.3 分钟**，包含 6 次初始验证训练和 162 次测试期周度重训，共保存 168 个检查点、2,400 条 epoch 记录。测试期同为 183 天、4,392 小时：

| 模型 | MAE | RMSE | 相对周基线 MAE 改善 | 负价 F1 |
| --- | ---: | ---: | ---: | ---: |
| 前一周同小时 | 114.372 | 174.502 | 0.00% | 0.4007 |
| LEAR | 83.621 | 124.204 | 26.89% | 0.0067 |
| LightGBM | 81.777 | 127.062 | 28.50% | 0.1932 |
| LSTM（三种子集成） | 80.565 | 125.470 | 29.56% | 0.2170 |
| TFT（三种子集成） | **79.771** | 125.461 | **30.25%** | 0.3589 |

TFT 集成相对 LightGBM 的 MAE 降低 **2.45%**，LSTM 集成降低 **1.48%**。单种子 MAE 均值±总体标准差为 LSTM `81.055 ± 0.483`、TFT `80.886 ± 0.260`。本轮 TFT 集成的整体 MAE 最低，LEAR 的 RMSE 仍最低；TFT 负价识别优于另外三个学习模型，但仍未超过周基线。总体误差改善不代表交易收益。19 项离线测试通过，全部训练截止边界、预测覆盖和 6 个末次检查点的恢复预测均已核对。

### 4.4 Chronos-2 / Timestral / TiRex-2：零样本测评

官方模型适配器见 [foundation_models.py](../../scripts/foundation_models.py)，统一入口见 [run_foundation_models.py](../../scripts/run_foundation_models.py)，配置见 [foundation_models.json](../../configs/foundation_models.json)。新模型需要 Python 3.12 与 NumPy 2，使用独立环境，避免改变原有基线环境：

```bash
uv venv --python 3.12 .venv-foundation
uv pip install --python .venv-foundation/bin/python -r requirements-foundation-lock.txt
.venv-foundation/bin/python -m unittest discover -s tests -v
.venv-foundation/bin/python scripts/run_foundation_models.py

# 中断后按原配置、源数据与代码快照恢复；已完成模型的结果会被校验并复用
.venv-foundation/bin/python scripts/run_foundation_models.py --resume

# 已下载并完成一轮后，可断网复用固定版本缓存
HF_HUB_OFFLINE=1 .venv-foundation/bin/python scripts/run_foundation_models.py --resume
```

首次运行从官方 Hugging Face 仓库下载 `amazon/chronos-2`、`mistralai/Timestral-209M`、`NX-AI/TiRex-2`，保存精确 revision 和权重 SHA-256；缓存位于被 Git 忽略的 `data/model_cache/`。重新开展独立测评请用新的 `--output`。`--prepare-only` 只下载与审计，不评分，随后使用 `--resume` 继续。默认 CPU、2 线程，固定 168 小时历史、24 小时预测，不微调、不按测试结果调参；统一取 q50 点预测，并保存 q10/q90 区间。依赖锁同时固定 Timestral 官方源码提交。

本任务实际为**单个价格目标 + 8 维日历协变量**；多变量能力由官方模型提供，本轮不额外构造未经核验的供需预测目标。未来协变量仅含日历，不使用 ERA5、供需字段、实时价格或目标日真实价格。缺失历史直接拒绝，不填补；Chronos 明确关闭跨任务交叉学习，三个模型都通过单样本与混入较晚日期的批量预测一致性检查，防止批量预测跨日期泄漏。

同一山东源数据与 2025-07-01 至 2025-12-30 的 4,392 点上，实测如下。既有模型复用已保存预测，核对源文件哈希、日期/小时、真实标签及预测时点后逐行评分；基础模型零样本与原有监督模型训练方式不同。

| 模型 | MAE | RMSE | 负价 F1 |
| --- | ---: | ---: | ---: |
| 前一周同小时 | 114.372 | 174.502 | 0.4007 |
| LEAR | 83.621 | 124.204 | 0.0067 |
| LightGBM | 81.777 | 127.062 | 0.1932 |
| LSTM（三种子集成） | 80.565 | 125.470 | 0.2170 |
| TFT（三种子集成） | 79.771 | 125.461 | 0.3589 |
| Chronos-2 | 91.906 | 139.143 | 0.2978 |
| Timestral-209M | 86.475 | 131.006 | 0.3395 |
| TiRex-2 | 86.677 | 130.930 | 0.2974 |

Timestral 在本轮三个基础模型中 MAE 最低，但仍比 LightGBM 高 5.75%；三个基础模型的尖峰 MAE 均低于 LightGBM。完整报告见 [foundation_models_v1.md](../../reports/foundation_models_v1.md)，指标见 [foundation_models_v1_metrics.json](../../reports/foundation_models_v1_metrics.json)。逐小时预测、输入截止日期审计、分组指标、固定权重版本、区间覆盖率与对比图保存在 `outputs/foundation_models_v1/`。本轮共 23 项离线测试通过。

**回测边界：**沿用 D-1 12:00 已知 D-1 完整日前价格、目标日日历提前可得的假设，源发布时间仍待核验。权重发布晚于测试区间，公开预训练语料不足以排除山东数据污染；本地特征与评分无前视不代表预训练污染已排除，不能将本轮称为已完全核实的无泄漏交易回测。

### 4.5 Timestral 四种规模对比

配置见 [timestral_sizes.json](../../configs/timestral_sizes.json)。沿用上一轮 168 小时价格历史、8 维日历协变量、CPU 2 线程与 batch 8，不调参、不微调；四个规模均实际运行，同一 183 天、4,392 点评分。209M 全部分位数预测与前轮逐值完全一致。

| Timestral 规模 | MAE ↓ | RMSE ↓ | 负价 F1 ↑ | 整轮推理秒数 ↓ |
| --- | ---: | ---: | ---: | ---: |
| 5M | 87.763 | 132.517 | 0.2771 | 1.10 |
| 28M | **85.594** | 132.012 | 0.3198 | 3.63 |
| 123M | 85.939 | **130.744** | 0.3389 | 12.79 |
| 209M | 86.475 | 131.006 | **0.3395** | 20.36 |

本轮 28M 的 MAE 比 209M 低 1.02%，整轮推理约快 5.60 倍，是后续轻量实验候选；123M 的 RMSE 最低。28M 与 123M 的 MAE 差距仅约 0.345，不能据此认定稳定优胜；四者总体 MAE 都高于现有 LightGBM（81.777）。计时不含下载/加载，含批量隔离检查和少量结果整理，不代表单请求延迟。

```bash
.venv-foundation/bin/python scripts/run_foundation_models.py --config configs/timestral_sizes.json --output outputs/timestral_sizes_v1
.venv-foundation/bin/python scripts/summarize_timestral_sizes.py
```

完整报告见 [timestral_sizes_v1.md](../../reports/timestral_sizes_v1.md)，逐小时预测、区间覆盖率、分组指标、权重版本与审计记录见 `outputs/timestral_sizes_v1/`。四种规模均通过批量隔离检查，23 项离线测试通过。沿用前轮价格/日历可得性假设及预训练污染未排除的限制；这是同一测试集上的后续规模探索，正式选型需在未看过的后续区间确认。

### 4.6 Timestral-28M：预测头与 LoRA 微调

实现见 [timestral_finetune.py](../../scripts/timestral_finetune.py)，入口见 [run_timestral_finetune.py](../../scripts/run_timestral_finetune.py)，固定配置见 [timestral_finetune.json](../../configs/timestral_finetune.json)。复用独立 `.venv-foundation` 和已下载的 28M 权重。

预测头训练 1,880,064 个参数；LoRA 在时间/变量注意力的 wQKV、wO 注入 rank=8、alpha=16，训练 276,480 个参数，其余冻结。统一输入 168 小时价格历史和 8 维日历协变量，以原生 21 分位 Pinball Loss 监督未来 24 点，关闭非负截断与极端值 cap，保留符号对称平均。

2024 年 359 日初始训练，2025 年 1–3 月 90 日选轮数（Head 9、LoRA 14），4–6 月 91 日确认但不改方案。测试起点使用全部已到达的 540 日历史；之后每 7 日继续训练，最近 56 个日历日回放 1 epoch。每次训练严格使用截止日之前的标签；增量回放与既有基线全历史重训的方式不同。

| 模型 | MAE ↓ | RMSE ↓ | 负价 F1 ↑ |
| --- | ---: | ---: | ---: |
| WeeklyNaive | 114.372 | 174.502 | 0.4007 |
| LEAR | 83.621 | 124.204 | 0.0067 |
| LightGBM | 81.777 | 127.062 | 0.1932 |
| LSTM | 80.565 | 125.470 | 0.2170 |
| TFT | 79.771 | 125.461 | 0.3589 |
| ZeroShot | 85.594 | 132.012 | 0.3198 |
| Head | 83.412 | 129.365 | 0.1538 |
| LoRA | 84.550 | 131.116 | 0.1910 |

Head/LoRA 为 42、43、44 三个种子的等权平均；单种子、确认指标、概率区间与完整训练/推理成本见[微调实测报告](../../reports/timestral_finetune_v1.md)。30 项离线测试通过；报告生成时重新计算全部指标，并从首末日的 12 个历史检查点独立回放预测。

```bash
.venv-foundation/bin/python scripts/run_timestral_finetune.py --output outputs/timestral_finetune_repeat
# 中断恢复：配置、源数据、执行代码必须一致
.venv-foundation/bin/python scripts/run_timestral_finetune.py --output outputs/timestral_finetune_repeat --resume
.venv-foundation/bin/python scripts/report_timestral_finetune.py --output outputs/timestral_finetune_repeat
```

完整产物保存于 `outputs/timestral_finetune_v1/`，包括逐小时预测、21 分位数组、170 个训练截止点检查点、训练日志及审计信息。源价格发布时间与日历提前可得性仍依赖前述假设，预训练污染未完全排除；同一测试集已参与先前规模探索，正式独立确认需要新的未见区间。

**前轮口径更正：**Timestral 先前规模测评沿用了默认条件非负截断（历史全非负时将负预测截为 0），此前“不裁剪负价”的概括仅适用于原始标签。当前零样本和两条微调路线均关闭该后处理，收益相对此同设置零样本计算。

## 5. 评估与复现

- **山东划分计划**：按时间先后划分训练、验证与测试，覆盖完整季节；日前有效标签先完成小时口径审计，具体边界在训练前锁定，禁止随机拆分时间序列。
- **滚动回测**：按交付日生成预测；验证期确定重训练频率与窗口，测试期严格按既定方案执行，每次仅使用当时可得的数据。
- **核心指标**：MAE、RMSE，以及相对前一周同一时段基线的 MAE 比值；所有比值使用相同有效样本，分母为零时单独处理。避免将 MAPE 作为主指标，以适应零价和负价。
- **专项评估**：按小时、季节、工作日/节假日、尖峰和负价时段报告误差；事件阈值依据市场规则或训练集预先确定。
- **概率预测**：报告 Pinball Loss、区间覆盖率及区间宽度。
- **实验记录**：保存配置、数据校验值、代码版本、随机种子、模型权重、逐时预测和指标；深度模型使用多个随机种子报告波动。

首期验收要求是完整复现数据处理与回测、与基线公平比较并解释误差。若模型未优于基线，保留结果并分析原因，不预先承诺精度或收益。

## 后续工作

核验日前标签和特征发布时间，补充带起报时间的历史天气预报，再开展供需、天气和历史窗口消融。

交给出价模块的预测至少应标识预测时点、交付时段、价格单位、模型和数据版本，以及点预测或分位数。历史预测的可得性仍受上述实验假设约束。
