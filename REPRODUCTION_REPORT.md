# TabularMark 论文复现报告

## 摘要

本文复现 CCS 2024 论文 **TabularMark: Watermarking Tabular Datasets for
Machine Learning**。复现工作在 `hustserver65` 的专用 Docker 容器中完成，核心
XGBoost 实验使用两张 NVIDIA A100。最终一次 `--stage all` 全流程运行耗时 130.99 秒，
连续生成论文 Tables 2-27 和 Figures 6-11 的对应 CSV/PNG 结果。

从实验覆盖范围看，论文所有编号实验均已有对应输出；从协议一致性看，23 张表按论文
或作者公开代码协议重建，Table 12、Table 15、Table 27 因作者未公开完整实现细节而采用
明确标注的替代协议。复现结果支持论文关于可检测性、非侵入性、基础攻击鲁棒性和参数
权衡的主要结论，但不能宣称所有表格数值都与论文一致。Table 27 的 Shapley 优势没有
复现，Table 12 和 Table 15 的代理结果不能作为原协议的等价验证。

## 1. 为什么选择复现这篇论文？

选择 TabularMark 主要有五个原因。

1. **研究问题有实际意义。** 表格数据可以被近乎零成本复制，并直接用于训练机器学习
   模型。如何证明共享数据的所有权，是数据安全、数据交易与模型训练中的现实问题。
2. **论文质量和新颖性较高。** 论文发表于 CCS 2024，将表格数据水印转化为统计假设
   检验问题，方法简单、可解释，且同时考虑了数值属性和类别属性。
3. **评价体系完整。** 论文围绕 detectability、non-intrusiveness、robustness、与现有
   方法比较、参数权衡五个研究问题给出了大量表格和曲线，适合做可核验的论文复现。
4. **公开了核心代码和数据。** Synthetic、Forest Cover Type、HOG、Boston Housing
   及 Figure 7 的比较代码均能获得。虽然 artifact 不完整，但足以先恢复核心实验。
5. **一天内形成高完成度的可能性较大。** 该方法不需要训练大型神经网络，实验规模也
   不大。A100 可以加速 XGBoost，但真正的工作重点是统一路径、修复 Notebook、固定
   随机性并核对实验协议，因此比依赖超大模型和长时间训练的论文更适合本任务。

## 2. 论文主要解决的问题是什么？

论文要解决的问题是：**如何为机器学习使用的表格数据嵌入所有权水印，同时保证水印
容易检测、不明显破坏下游模型效用，并能抵抗修改、插入和删除等攻击。**

现有关系数据库水印方法常依赖主键、最低有效位、整数属性或数据统计量，难以同时支持
浮点和类别属性，也很少直接衡量水印对下游模型性能的影响。TabularMark 的核心流程为：

1. 数据拥有者使用秘密 seed 从目标属性中选择少量 key cells。
2. 对每个 key cell，将允许的扰动范围划分成多个小区间，再随机分为 green domains
   和 red domains。
3. 嵌入时只从 green domains 选择扰动，因此水印数据的偏差会有意集中在 green
   domains；无水印数据的偏差按论文假设应近似随机。
4. 检测时用同一 seed 恢复 domains，统计落入 green domains 的 key cells，并用单样本
   比例 z 检验判断水印是否存在。默认阈值 1.96 对应 5% 显著性水平。
5. 行序被插入或删除攻击破坏时，使用多个属性的高位信息匹配原始 tuple 和可疑 tuple。

论文追求三个核心性质：水印能可靠检出；水印数据仍能训练出性能接近原始数据的模型；
攻击者只有大幅破坏数据、同时牺牲模型效用，才能有效擦除水印。

## 3. 复现结果如何？

### 3.1 完成范围与可信度分级

| 研究部分 | 论文结果 | 当前状态 | 协议说明 |
|---|---|---|---|
| RQ1 可检测性 | Table 2、Figure 6 | 完成 | 论文协议及额外 100% 擦除压力测试 |
| RQ2 非侵入性 | Tables 3-5 | 完成 | Forest、HOG、Boston、Zoo |
| RQ3 基础攻击 | Tables 6-11 | 完成 | alteration、insertion、deletion |
| RQ3 数据清洗 | Table 12 | 代理完成 | 使用透明的约束修复代理，不等同于旧版 HoloClean 推断 |
| RQ3 匿名化 | Tables 13-14 | 完成 | Iris、3-anonymization |
| RQ3 数据生成 | Table 15 | 代理完成 | 类条件高斯生成器，不等同于论文未公开的语言模型生成器 |
| RQ3 更多数据 | Tables 16-18 | 完成 | HOG、Boston、Zoo alteration |
| RQ4 横向比较 | Figure 7 | 完成 | 从作者 HistMark/SemMark/TabularMark Notebook 重建 |
| RQ5 参数权衡 | Tables 19-24、Figures 8-10 | 完成 | `p`、`nw`、`gamma` 的全部网格和无攻击基线 |
| Discussion 噪声选择 | Tables 25-26 | 完成 | 均匀噪声与截断正态噪声 |
| Discussion Shapley | Table 27 | 尝试完成 | 16-permutation TMC-Shapley；论文未公开估计细节，结论未复现 |
| Discussion 匹配 | Figure 11 | 完成 | tuple-by-tuple 与排序后二分检索 |

因此，“所有实验都跑了”应准确理解为：**所有编号图表已有服务器运行输出，但三项缺失
原始实现的实验采用了明确标注的重建/代理协议。** 这三项不能与原协议结果等价看待。

### 3.2 可检测性：Table 2 与 Figure 6

| 数据集 | 论文 Dw z-score | 复现 Dw z-score | 判断 |
|---|---:|---:|---|
| Synthetic | 17.3 | 17.3205 | 非常接近 |
| Forest Cover Type | 18.6 | 20.0000 | 均远高于 1.96 |
| HOG | 12.3 | 12.2474 | 非常接近 |
| Boston Housing | 6.91 | 7.0711 | 接近 |

四个水印数据集的 z-score 都显著超过阈值 1.96，水印可检测性复现成功。Forest 的差异
与作者 watermark 脚本使用 300 个 key cells、检测脚本和论文使用 400 个 key cells 的
冲突有关。

| ROC 数据集 | 论文 AUC | 论文邻域协议复现 | 含 100% 擦除的压力测试 |
|---|---:|---:|---:|
| Synthetic | 0.93 | 0.8477 | 0.7000 |
| Boston Housing | 0.94 | 0.9135 | 0.6824 |

Boston 的论文协议 AUC 接近原文，Synthetic 低 0.0823。严格压力测试表明，是否包含
100% 擦除样本会显著影响 AUC，因此单独报告一个 AUC 不能完整描述检测器鲁棒性。

### 3.3 非侵入性：Tables 3-5

Forest Cover Type 的三个目标类别 F1 中，复现的 Do/Dw 差异均很小；绝对值更接近作者
Notebook 保存输出，而不是论文表格。Boston Housing MSE 复现为
`23.158586 / 24.304466`，与作者 Notebook 保存输出逐位一致。

HOG Accuracy 为 `0.7944 / 0.8204`，没有达到论文的 `0.942 / 0.940`，但作者 Notebook
自身保存的 Do 也只有 `0.8167`，说明这里存在明显的 paper/artifact gap。Zoo 的水印
z-score 为 `3.8730`，与论文 `3.87` 几乎完全一致；Do/Dw Accuracy 均为 `0.9355`，
高于论文的 `0.900 / 0.870`，仍支持水印没有明显降低效用。

### 3.4 鲁棒性：Tables 6-18

- Forest alteration 的复现 z-score 为
  `14.20, 9.70, 7.04, 0.12, -2.08`，论文为
  `13.8, 8.67, 3.68, -1.11, -6.05`。两者均在 80% alteration 时跌破 1.96。
- 100% Forest alteration 后，三个类别的 F1 降至
  `0.2394, 0.0087, 0.0505`，成功复现“擦除水印时模型效用严重下降”。
- insertion 的所有 z-score 仍高于 1.96；deletion 的 z-score 随删除比例总体下降，
  100% 删除没有剩余 tuple，统计量按定义为 NaN。这与论文结论一致。
- Iris 3-anonymization 中，匿名列从 1 增至 3 时，z-score 从 `4.02` 降至 `1.34`，
  Accuracy 从 `0.978` 降至 `0.822`。论文对应结果为 `5.85 -> 1.10` 和
  `0.911 -> 0.786`，方向一致。
- HOG alteration 的前三个 z-score 为 `9.96, 7.02, 4.90`，与论文
  `9.71, 7.13, 4.78` 很接近；绝对 Accuracy 仍受前述 HOG artifact 差异影响。
- Boston 与 Zoo alteration 均复现了攻击增强后检测能力与模型效用下降的总体趋势，
  但小数据集上的单 seed 指标存在波动。

Table 12 的约束修复代理会把 Adult/Hospital 水印 z-score 大幅降到零附近，与论文的
HoloClean 结果不同。这不能证明论文错误，因为代理修复比 HoloClean 更确定、更激进；
它只说明清洗攻击结果高度依赖具体修复器。Table 15 的高斯生成代理得到 Dw' Accuracy
`0.9556`，没有复现论文语言模型生成器的 `0.778`，也不应作等价比较。

### 3.5 横向比较：Figure 7

TabularMark 的七个类别 F1 与 Original 基本重合。HistMark 在类别 2 和 6 上分别只有
`0.0304` 和 `0.0852`，明显低于 Original 的 `0.6135` 和 `0.7732`；SemMark 介于两者
之间。攻击比例从 0 增至 60% 时，三种方案的 mismatch percentage 总体上升。

这组结果支持论文的主要比较结论：TabularMark 的模型效用损失更小；HistMark 的
histogram shifting 会对类别属性造成更强的数据失真。复现还暴露了一个评估问题：作者
Notebook 在不同方案中对 primary key 特征的处理不完全一致，本次统一排除 primary key，
避免把无语义的行号当作模型特征。

### 3.6 参数权衡：Tables 19-24 与 Figures 8-10

全部 25 个 `p x attack`、25 个 `nw x attack` 和 25 个 `gamma x attack` 点已生成。
相对趋势与论文一致：增大 `p` 或 `nw` 通常提高相同攻击强度下的 z-score；减小
`gamma` 通常提高 z-score，但可能增加误报风险。水印前后 Accuracy 的差异较小。

复现 Accuracy 约为 `0.85-0.88`，低于论文的 `0.87-0.98`，但更接近作者 Notebook
保存的 `0.8533-0.8733`。Table 24 的 Do z-score 对单 seed 很敏感，未逐项复现论文的
多次平均值，进一步说明误报分析应报告多 seed 的均值和置信区间。

### 3.7 Discussion：Tables 25-27 与 Figure 11

Table 25 是本次最清晰的新增成功结果之一：Boston MSE 为
`Do=23.1586, Uniform=24.3045, Normal=23.3470`。正态分布选择使水印数据更接近原始
模型效用，支持论文关于“小扰动应有更高抽样概率”的结论。

Table 26 的 z-score 下降方向与论文一致，但 MSE 并不严格单调，说明 50 个 key cells 的
小样本设置对随机 seed 较敏感。

Table 27 没有复现：16-permutation TMC-Shapley 得到
`Random=28.8721, Shapley=32.2542`，而论文为 `24.1 / 23.3`。作者没有公开 Table 27
代码、验证集定义、Shapley 估计器或采样次数，因此当前结果只能定性为“公开信息不足下
的复现失败”，不能通过调参反向拟合论文结论。

Figure 11 中，两种匹配方法在所有插入比例下均得到完全相同的 z-score `7.0711`；二分
检索耗时约 `0.024-0.032` 秒，逐 tuple 检索约 `0.028-0.042` 秒。检测结果不变且二分
检索更快，论文结论复现成功。绝对时间低于论文，是因为本次实现消除了 Notebook I/O
和重复 DataFrame 操作，绝对耗时不宜直接横向比较。

### 3.8 总体判断

若判断标准是“论文核心结论和主要趋势是否成立”，本次复现总体成功；若判断标准是
“所有单元格逐值相同”，则不能认为完全复现。最可靠的一致结果是 Table 2、Zoo
z-score、Boston Notebook MSE、HOG 前三档攻击 z-score、Figure 7 的相对关系以及
Table 25 的正态噪声改进。最需要在答辩中主动说明的是 HOG 绝对 Accuracy、Synthetic
ROC、三项替代协议和 Table 27 的失败结果。

## 4. 复现过程中论文没有提到的新发现

### 4.1 论文、代码和 Notebook 保存输出是三种不同证据

1. Forest watermark 脚本使用 300 个 key cells，检测脚本和论文使用 400。
2. `watermark_synthetic.py` 在 key-cell 循环内重复执行 `temp = origin.copy()`，实际
   只保留最后一个 key cell 的扰动。
3. insertion/deletion Notebook 把 `set` 当 NumPy 索引使用，并将匹配行号与 seed
   直接 zip，原始代码无法稳定执行。
4. 多处路径写死为 `/home/zhengyihao/TabularMark/...`。
5. 仓库没有正式依赖文件，多个 Notebook 保存了报错或中断输出。
6. HOG Accuracy、Forest F1、Synthetic Accuracy 和 ROC 样本构造在论文与公开
   artifact 之间存在明显差异。

统一入口只修复了执行必需的问题，并固定随机种子；没有使用论文表值反向拟合结果。

### 4.2 AUC 高度依赖样本构造协议

是否加入 100% 擦除样本，会使 Synthetic AUC 从 `0.8477` 降至 `0.7000`，Boston 从
`0.9135` 降至 `0.6824`。ROC/AUC 报告应同时说明正负样本构造、攻击强度分布和分层
结果，否则一个较高 AUC 可能掩盖极端攻击下的失效。

### 4.3 算力不是主要瓶颈

两张 A100 使全阶段只需 131 秒。真正消耗时间的是核对 paper/artifact、补齐数据、修复
Notebook 和恢复协议。GPU 不能消除论文与公开 artifact 的不一致，可复现性首先取决于
数据版本、代码、随机种子和评价定义。

### 4.4 替代协议可能改变论文结论

约束修复代理比 HoloClean 更容易擦除水印；高斯生成代理又比论文语言模型生成器保留了
更多效用；TMC-Shapley 没有带来论文宣称的收益。这说明数据清洗器、生成器和数据估值器
不是可随意替换的实现细节，而是决定实验结论的重要组成部分。

### 4.5 非侵入性需要同时报告绝对值和相对变化

HOG 的绝对 Accuracy 没有达到论文值，但 Dw 没有比 Do 下降。如果只看绝对指标，会忽略
水印前后的相对效果；如果只看相对变化，又会掩盖 paper/artifact gap。更合理的方式是
同时给出论文值、作者 artifact 值、复现值和 Do-Dw 差值。

## 5. 论文带来的启发

1. **数据水印可以转化为统计假设检验。** 不必嵌入可逐位读取的字符串，也可以人为制造
   只有密钥持有者能验证的统计偏差，并显式控制理论误报率。
2. **安全性和模型效用必须联合评价。** 水印更强通常更容易检测，也更可能破坏数据；
   `p`、`nw`、`gamma` 展示了这种不可避免的权衡。
3. **攻击成功不能只看水印是否消失。** 如果攻击者必须把数据改到模型几乎不可用才能
   擦除水印，数据拥有者仍实现了保护目的。z-score 与下游模型效用应联合报告。
4. **协议透明度与数值同样重要。** 多 seed、置信区间、数据拆分、攻击样本组成和实现
   版本都应成为正式 artifact 的一部分，而不是只保留在作者环境中。
5. **仍有清晰的后续研究方向。** TabularMark 是非盲水印，检测时需要原始数据；未来可
   研究盲检测、按特征/样本重要性自动选 key cells、更强攻击模型，以及标准化的多 seed
   误报与鲁棒性评估。

## 6. 实验环境与运行方法

- 服务器：`hustserver65`
- Docker 容器：`tabularmark-repro`
- GPU：NVIDIA A100-SXM4-80GB + NVIDIA A100-SXM4-40GB
- Python：3.10.21
- XGBoost：1.7.6
- 固定随机种子：10000
- 依赖来源：清华 Conda 镜像、阿里云 PyPI 镜像
- 完整全阶段耗时：130.99 秒

在服务器容器的项目目录执行：

```bash
.conda-env/bin/python reproduce.py --stage all --xgb-device gpu
```

也可以分别执行 `core`、`robustness`、`extended`、`comparison`、`tradeoffs` 和
`discussion` 阶段。每个阶段均会生成独立的 `run_metadata_*.json`。

## 7. 结果清单

主要表格结果位于 `outputs/`：

- `table2_detectability.csv` 至 `table27_shapley.csv`
- 合并编号表：`tables8_9_insertion.csv`、`tables10_11_deletion.csv`、
  `tables23_24_gamma_baselines.csv`
- Figure 6 数据与图片：`figure6_roc_data.csv`、`figure6_roc.png`
- Figure 7 数据与图片：`figure7_f1.csv`、`figure7_mismatch.csv`、
  `figure7_comparison.png`
- Figures 8-10：`figure8_p_tradeoff.png`、`figure9_n_tradeoff.png`、
  `figure10_gamma_tradeoff.png`
- Figure 11：`figure11_matching_data.csv`、`figure11_matching.png`
- 完整运行元数据：`run_metadata_all.json`

所有 PNG 均为 2200 x 924，已完成目视检查，无截断、重叠或缺失图例。

## 8. 最终结论

本次复现已经达到“所有编号实验均有结果、核心结论得到验证、差异能够解释、代码可在
服务器 Docker 内一键重跑”的交付状态。需要保留的边界是：Table 12、15、27 缺少作者
原始实现细节，前两项是透明代理，后一项未复现论文优势。报告这些差异比隐藏失败结果或
用论文值补表更符合论文复现的目标。
