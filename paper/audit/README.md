# HCN clipped 实验数据包

先阅读 `HCN_Formal_Experiment_Audit_20261003.md`。

本包统一处理有限实数尾概率：`p_reported = min(1, max(0, p_raw))`，使用完整精度原始预测重算相对误差。原始文件不修改。NaN/Inf 保持不可用，不静默裁剪或删除。

## 数据范围

- 新上传正式包：2D product、2D Harrison、10D non-product 的 seed12345、20D 三 seed 固定25k结果，以及30D H100 profile。
- 先前 draft 对应来源：30D product 固定50k、50D product 固定100k，以及50D同一条轨迹的 late-training diagnostics。
- 不纳入10D另两个seed，不检查或纳入30D non-product。
- 主结果始终使用指定终点。trajectory、exact-transform diagnostic、independent-marginal comparator 均单独标记，不当作独立神经网络运行。

## 复算

在解压后的目录执行：

```bash
python audit_results.py
```

只需 Python 3 标准库；无需 GPU、PyTorch、网络或 checkpoint。脚本先验证源数据哈希，再生成 `results/` 和核对报告。它会覆盖本包内的派生结果和报告，不会修改 `inputs/`。

## 结果列

- `raw_prediction`：原始 CSV 中的实数文本；不覆盖。
- `clipped_prediction`：有限实数投影结果。
- `*_relative_error_percent` / `*_mre_percent`：百分数，例如 `6.5` 表示6.5%。
- `clipped_sample_sd_pp`：各seed区间MRE的样本标准差，单位百分点；单次运行留空。
- `role`：区分主模型、激活对照、基线、反演诊断、同轨迹稳定性及独立边际比较。
- `degree=0`：仅表示无需 Talbot 反演的解析 independent-marginal comparator。
- `raw_invalid`：非有限或超出 `[0,1]`；0与1本身属于合法闭区间。
- `source_*`：可追溯的文件、行号、来源类别与 SHA-256。

`selection_manifest.json` 保留源数据路径和身份核对记录。其 checkpoint/AST 核对在整理阶段执行，包内不复制模型权重；日后运行 `audit_results.py` 只复算数值表，不重新执行那些代码检查。

10D参考值仍属于pilot；profile全训练时间属于外推。具体解释见核对报告。
