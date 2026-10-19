# HCN 实验代码与 clipped 结果核对（2026-10-03）

本次整理以新上传的 `formal experiment.zip` 为主。按作者补充要求，30D/50D product-form 使用先前 draft 对应的固定终点逐点 CSV；这些 CSV 的 raw MRE 与旧稿表 6、7 一致。所有计算使用 CSV 的完整精度，不从旧稿已舍入的 MRE 反推。

## 1. 采用的规则

- 所有有限实数尾概率统一投影到 `[0,1]`，随后逐点重算相对误差和区间 MRE；保留原值、原始越界标记及来源。
- NaN/Inf 不转成 0 或 1。若某报告区间含非有限点，该区间 MRE 标为不可用，不通过删除失败点得到均值。本次纳入的逐点记录全部有限。
- 10D non-product 只纳入 seed 12345 的 step 25000；不汇总另两个 seed。它是单次 pilot 展示，不提供跨 seed 稳健性证据。30D non-product 未纳入，也未检查其内容。
- 20D 按三个 seed 的固定 step 25000 分别算 MRE，再报告均值及样本标准差。三个重复评估目录中的对应 CSV 完全相同，只计一次。
- 保留所有已有 degree 的结果，主规则维持 2D Degree 5、高维 Degree 8；不根据本次重算选择更好的 degree 或 checkpoint。

公式：`MRE (%) = 100 × mean(|clip(p_raw,0,1) − p_ref| / p_ref)`。表内 SD 单位为百分点，采用 `ddof=1`；单次运行不报告 SD。

## 2. 主要结果

| 实验/方法 | Degree | 运行数 | MRE 1:5 (%) | MRE 1:8 (%) | MRE 1:10 (%) |
|---|---:|---:|---:|---:|---:|
| 2D product / HCN | 5 | 1 | 2.147 | 2.213 | 9.737 |
| 2D product / NN4MGF published | 5 | 1 | 2.448 | 8.050 | 13.915 |
| 2D Harrison / HCN | 5 | 1 | 1.422 | — | — |
| 2D Harrison / NN4MGF published | 5 | 1 | 6.684 | — | — |
| 10D non-product / HCN（pilot） | 8 | 1 | 1.710 | 3.684 | — |
| 10D / independent-marginal comparator | — | 1 | 8.510 | 19.409 | — |
| 20D product / HCN，三 seed 均值 | 8 | 3 | 1.441 | 7.632 | 12.264 |
| 20D product / NN4MGF archived | 8 | 1 | 2.802 | 8.148 | 18.908 |
| 20D product / complex tanh，三 seed 均值 | 8 | 3 | 8.604 | 22.105 | 37.036 |
| 30D product / HCN，同 degree 比较 | 5 | 1 | 3.429 | 10.107 | 19.084 |
| 30D product / NN4MGF published | 5 | 1 | 8.353 | 10.476 | 21.507 |
| 30D product / HCN，主结果 | 8 | 1 | 1.683 | 6.546 | 8.872 |
| 50D product / HCN，主结果 | 8 | 1 | 1.495 | 6.367 | 8.011 |

20D HCN Degree 8 的 1:8 结果为 **7.632% ± 1.565 个百分点**（三个 seed 的均值 ± 样本 SD）。这里使用25k固定终点，不混入旧探索中的精修、五 seed 或最佳 checkpoint 数字。

10D 的参考数组为 `[0.97204, 0.72091, 0.40030, 0.19124, 0.08443, 0.03576, 0.014645, 0.005905]`。文件将其标为未经认证的 screenshot pilot，无经验证的离散误差界；因此这里只能称为相对 pilot reference 的误差。

2D/30D NN4MGF 使用已发布表中的有限精度数值；20D 使用归档的 released-checkpoint 全精度评估，未重新运行 NN4MGF，且此处不声称相同训练预算。

## 3. Clipping 对重点结果的影响

| 实验 | Degree | 阈值 | raw | clipped |
|---|---:|---:|---:|---:|
| 10D_nonproduct | 8 | 1 | 99.4714522362 | 1 |
| 30D_product | 8 | 1 | 1.00010762095 | 1 |
| 50D_product | 8 | 2 | 1.00047763932 | 1 |

| 实验 | raw MRE 1:8 (%) | clipped MRE 1:8 (%) | 原始越界点 / 8 |
|---|---:|---:|---:|
| 10D_nonproduct | 1269.982750 | 3.683881 | 1/8 |
| 30D_product | 6.547216 | 6.545871 | 1/8 |
| 50D_product | 6.373004 | 6.367030 | 1/8 |

30D/50D 主结果都有轻微 overshoot，而非完全没有越界。Clipping 对它们的整体 MRE 影响很小。10D 的主要变化来自 t=1 的大幅越界；clipping 将报告值投影回合法区间，但不代表修复了产生该 raw 输出的数值机制。

20D complex-tanh 的 seed 22345 / Degree 8 原始 MRE 非常大，clipping 后可成为有限的区间结果。按统一规则应保留该 seed 并重算三 seed 汇总；旧稿中“只对两个成功 tanh seed 求均值”的叙述需要更新。原始不稳定性由 raw 越界和原值单独呈现。

## 4. 代码与 Section 3 的对应

本次对新代码与 Section 3 写作所用代码进行了 31 项函数/类 AST 比对，所查单元一致，包括 Cauchy 激活、坐标共享/全局分支、BAR 和边界/实轴结构损失。没有发现需要改动 Section 3 核心网络或损失公式的新差异。该比对不等同于对全部实现或理论结论的形式化证明。

- 原点归一化和边界质量由结构硬约束实现；没有另加原点拟合损失。
- Cauchy 全纯性限定在无极点区域；去除 CR penalty 后，参数优化仍需自动微分。
- 10D 的 global interaction 分支进入内部与边界网络；边界分支屏蔽本面的坐标。2D Harrison 的全局扩展进入内部网络，不能把二者写成完全相同结构。
- 10D 采样实部下界为 −0.2，2D/product-form 家族为 −0.5；训练细节需要区别写明。
- 2D 每次更新 3416+427+427 个基础输入，15000 次更新合计 **64.05M**。10D、20D、30D、50D 对应固定终点合计 **64M**。
- 30D/50D 的历史结果绑定各自固定终点，不使用 smoke 输出代替正式结果。此次已将旧稿的逐点结果和 checkpoint 身份核对补齐。

## 5. 同卡 runtime 与显存

两份报告均来自 NVIDIA H100 NVL：先 warmup 10 次，再测量 200 次更新。

| 项目 | HCN | NN4MGF |
|---|---:|---:|
| 平均每次更新（秒） | 0.165113 | 1.601046 |
| 200 次更新测量时间（秒） | 33.023 | 320.209 |
| 峰值 allocated（GiB） | 7.805 | 77.815 |
| 峰值 reserved（GiB） | 8.414 | 81.428 |
| 每次更新基础输入数 | 1280 | 32768 |
| 对应完整更新数 | 50,000 | 300,000 |
| 按 profile 外推完整时间（小时） | 2.293 | 133.420 |

完整 schedule 的时间比约为 **58.18×（外推）**，allocated 显存比约为 **9.97×**。不能称为完整训练实测加速比，不能写成等 batch/等参数量的吞吐比较，也不能把全部差异归因于去除 CR penalty。

这里实际计时代码的 NN4MGF 每次 optimizer update 有两次 batch=16384 的采样/反向传播，基础输入为32768。按该代码路径的300k更新，预算为9.8304B，HCN的64M占约0.6510%。按旧稿引用的论文口径300k×16384，则为4.9152B，占约1.3021%。正文必须明确区分“论文报告预算”和“本次计时代码路径预算”，不能混写成同一个数。

## 6. 后续正文更新时的注意项

- 评估协议、图注、表注统一改为 clipped probability；附表保留 raw invalid count 和 raw 原值。不要只替换表内 MRE 而保留“without clipping”。
- 20D 激活消融按所有三个有限运行重算；不把 clipping 后的有限预测仍按旧 raw 成功规则剔除。
- exact-transform 数值反演诊断也同时提供 raw/clipped 汇总。解释数值机制时保留 raw 误差；有符号的 inversion/network/total 加法分解不可混用 clipped total 和 raw 分量。本包额外保留全 raw 分解表。
- 50D late-training 表使用同一条 seed-12345 轨迹，不算六个独立 seed；100k 固定终点在 trajectory 中再次出现，仅作为对照，不重复计入主结果。
- 本次没有训练或加载 checkpoint 做推理。已验证9份 checkpoint 文件的 SHA-256 身份；这只能确认文件一致，不能代替独立复现推理。

## 7. 数据与复算

`inputs/` 保存本次采用的原始 CSV/JSON，`selection_manifest.json` 记录原路径、SHA-256、去重关系、checkpoint 校验和代码比对。

`results/pointwise_clipped.csv`：逐点 raw/clipped、误差、状态与来源。

`results/per_run_summary.csv`：按 seed、终点、degree、区间汇总。

`results/aggregate_summary.csv`：跨 seed 均值与样本 SD。

`results/changed_predictions.csv`：全部被 clipping 改动的点。

`results/signed_error_components_raw.csv`：20D 全 raw 有符号误差分解。

`results/runtime_summary.json` 与 `results/validation.json`：资源统计及核对记录。

复算：解压完整数据包，在文件夹内执行 `python audit_results.py`。只需 Python 标准库；不需要 GPU、PyTorch 或重新训练。
