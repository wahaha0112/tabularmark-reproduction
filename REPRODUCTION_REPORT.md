# TabularMark 复现报告

## 结论

本次复现覆盖了作者公开数据和代码能够支持的主要定量实验：Table 2–4、
Table 6–11、Table 20、Table 22，以及 Figure 6、Figure 8–10。所有正式实验均在
`hustserver65` 的专用 Docker 容器 `tabularmark-repro` 中运行；XGBoost 实验使用
两张 A100，代码、配置和结果均保存在本仓库。

复现结果支持论文的三个核心结论：水印后的 z-score 显著超过 1.96；对正常模型
效用影响较小；只有强攻击才能消除水印，此时数据的模型效用也明显下降。部分论文
表值无法由公开仓库逐值重现，但复现结果与作者 Notebook 保存输出高度一致，说明
主要差异来自论文与公开 artifact 不一致，而不是运行环境。

## 环境与耗时

- 服务器：`hustserver65`
- 容器：`tabularmark-repro`
- GPU：NVIDIA A100-SXM4-80GB + NVIDIA A100-SXM4-40GB
- Python：3.10.21
- XGBoost：1.7.6
- 固定随机种子：10000
- 依赖来源：清华 Conda 镜像、阿里云 PyPI 镜像
- Core：17.6 秒
- Robustness：112.2 秒
- Trade-offs：8.2 秒

## 关键结果对照

### Table 2：可检测性

| 数据集 | 论文 Dw | 复现 Dw | 结论 |
|---|---:|---:|---|
| Synthetic | 17.3 | 17.3205 | 近似一致 |
| Forest Cover Type | 18.6 | 20.0000 | 均远高于阈值；作者代码存在 300/400 key-cell 冲突 |
| HOG | 12.3 | 12.2474 | 近似一致 |
| Boston Housing | 6.91 | 7.0711 | 近似一致 |

原始数据和随机扰动数据的单次 z-score 会随密钥种子波动；论文给出的是多次平均值，
公开代码只保留了单个 seed 的可执行路径。四个复现 Dw 均显著超过检测阈值 1.96。

### Table 3：Forest Cover Type F1

| 数据 | 指标 | 论文 | 作者 Notebook | 本次 GPU 复现 |
|---|---|---:|---:|---:|
| Do | Category 2 | 0.888 | 0.9055 | 0.9061 |
| Do | Category 4 | 0.940 | 0.8929 | 0.8915 |
| Do | Category 6 | 0.848 | 0.8612 | 0.8591 |
| Dw | Category 2 | 0.887 | 0.9078 | 0.9062 |
| Dw | Category 4 | 0.937 | 0.8906 | 0.8985 |
| Dw | Category 6 | 0.845 | 0.8486 | 0.8572 |

本次结果更接近作者仓库 Notebook 中保存的实际输出。Do 与 Dw 的差异很小，
非侵入性结论复现成功。GPU 和 CPU 的直方图构建算法不同，F1 存在约千分位差异。

### Table 4：更多数据集的模型效用

| 指标 | 论文 Do | 论文 Dw | 作者 artifact | 本次复现 Do | 本次复现 Dw |
|---|---:|---:|---:|---:|---:|
| HOG Accuracy | 0.942 | 0.940 | Do=0.8167 | 0.7944 | 0.8204 |
| Boston Housing MSE | 24.8 | 25.6 | 23.1586 / 24.3045 | 23.1586 | 24.3045 |

Boston Housing 与作者 Notebook 保存输出逐位一致。公开 HOG 数据和随机森林代码不能
产生论文的 0.94；作者 Notebook 自身保存的 Do 结果也是 0.8167，证实这里存在
paper/artifact gap。

### Figure 6：ROC

| 数据集 | 论文 AUC | 论文描述的邻域协议 | 含 100% 擦除的压力测试 |
|---|---:|---:|---:|
| Synthetic | 0.93 | 0.8477 | 0.7000 |
| Boston Housing | 0.94 | 0.9135 | 0.6824 |

Boston Housing 已接近论文值。Synthetic 的公开 Notebook 与论文文字对样本构造数量和
攻击比例描述不一致；本次同时报告论文描述协议和更严格的 100% 擦除协议，避免通过
删掉困难样本来人为抬高 AUC。

### Table 6–11：鲁棒性

- Table 6 的复现 z-score 为 `14.20, 9.70, 7.04, 0.12, -2.08`；论文为
  `13.8, 8.67, 3.68, -1.11, -6.05`。两者都在 80% alteration 时跌破 1.96。
- Table 7 成功复现“攻击越强、模型效用越差”的趋势，100% alteration 时三个目标
  类别的 F1 为 `0.2394, 0.0087, 0.0505`。
- 两属性 insertion 的 z-score 为 `7.85–9.01`，与论文 `5.73–8.23` 同量级；
  三属性 insertion 为 `15.47–16.05`，论文为 `17.8–18.3`。
- deletion 的 z-score 随删除比例单调下降；100% 删除后没有剩余 tuple，统计量按定义
  为 NaN。论文表中对应单元格同样以反斜线表示无结果。

### Table 20、Table 22 与 Figure 8–10

全部 25 个 `p × attack` 点、25 个 `nw × attack` 点和 25 个
`gamma × attack` 点均已生成，同时保存了 z-score、accuracy 和 PNG 曲线。z-score 的
相对趋势与论文一致：增大 `p` 或 `nw` 通常提高抗攻击能力；较小的 `gamma` 产生更高
的 z-score。accuracy 的绝对值约为 0.85，与作者 Notebook 保存的 0.8533–0.8733
一致，但低于论文表中的约 0.87–0.95。

## 公开 artifact 中发现的问题

1. Forest Cover Type 的 watermark 脚本使用 300 个 key cells，检测脚本和论文使用
   400，参数不一致。
2. `watermark_synthetic.py` 在 key-cell 循环内重复执行 `temp = origin.copy()`，导致
   文件实际只保留最后一个 key cell 的扰动。
3. insertion/deletion Notebook 把 `set` 当 NumPy 数组索引使用，且匹配后的行号与
   divide seed 直接 zip，不能稳定执行。
4. 多处路径写死为作者机器上的 `/home/zhengyihao/TabularMark/...`。
5. 仓库没有正式依赖文件，多个 Notebook 保存了报错或中断输出。
6. 论文与公开 Notebook 的 HOG accuracy、Forest F1、synthetic accuracy 和 ROC
   采样协议存在明显不一致。

统一入口 `reproduce.py` 修复了上述可执行性问题，固定随机种子，并保留严格压力测试
结果。修正属于工程复现所必需的最小改动，没有使用论文表值反向拟合实验输出。

## 结果文件

- `outputs/table2_detectability.csv`
- `outputs/table3_forest_f1.csv`
- `outputs/table4_more_datasets.csv`
- `outputs/figure6_roc_data.csv`、`outputs/figure6_roc.png`
- `outputs/table6_alteration_z.csv`
- `outputs/table7_alteration_f1.csv`
- `outputs/tables8_9_insertion.csv`
- `outputs/tables10_11_deletion.csv`
- `outputs/table20_p_tradeoff.csv`、`outputs/figure8_p_tradeoff.png`
- `outputs/table22_n_tradeoff.csv`、`outputs/figure9_n_tradeoff.png`
- `outputs/figure10_gamma_data.csv`、`outputs/figure10_gamma_tradeoff.png`
- `outputs/run_metadata_core.json`
- `outputs/run_metadata_robustness.json`
- `outputs/run_metadata_tradeoffs.json`

## 重跑方法

在服务器 Docker 容器的项目目录中执行：

```bash
.conda-env/bin/python reproduce.py --stage core --xgb-device gpu
.conda-env/bin/python reproduce.py --stage robustness --xgb-device gpu
.conda-env/bin/python reproduce.py --stage tradeoffs
```
