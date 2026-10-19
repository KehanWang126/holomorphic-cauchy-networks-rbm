#!/usr/bin/env python3
"""Rebuild the audit from the bundled, unmodified source files (Python 3 stdlib).

No training, checkpoint deserialization, network access, or model inference.
Run: python audit_results.py
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def number(row, names):
    for name in names:
        if name in row and row[name] != '':
            return float(row[name]), row[name]
    raise ValueError(f'Missing numeric field: {names}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root/'selection_manifest.json').read_text(encoding='utf-8'))
    origins = {f['file']:f for f in manifest['files']}
    for item in manifest['files']:
        actual = hashlib.sha256((root/item['file']).read_bytes()).hexdigest()
        if actual != item['sha256']:
            raise ValueError(f'Source hash mismatch: {item["file"]}')
    points, warnings = [], []
    for ds in manifest['datasets']:
        with (root/ds['file']).open(encoding='utf-8-sig', newline='') as f:
            for line, row in enumerate(csv.DictReader(f), start=2):
                p, ptext = number(row, ['raw_prediction','prediction','published_prediction'])
                ref, _ = number(row, ['true_value','reference','ground_truth_tail'])
                if not (math.isfinite(ref) and 0 < ref <= 1):
                    raise ValueError(f'Invalid reference at {ds["file"]}:{line}')
                t = float(row['t'])
                if ds['benchmark'].endswith('_product'):
                    exact = -math.expm1(ds['dimension']*math.log1p(-math.exp(-t)))
                    if not math.isclose(exact, ref, rel_tol=2e-10, abs_tol=2e-12):
                        raise ValueError(f'Product reference mismatch at {ds["file"]}:{line}')
                method, role = ds['method'], ds['role']
                if method == 'from_row':
                    if row['method'] not in {'Cauchy','tanh'}:
                        raise ValueError(row['method'])
                    method = 'HCN_Cauchy' if row['method']=='Cauchy' else 'Complex_tanh'
                    role = 'model' if row['method']=='Cauchy' else 'activation_control'
                seed = str(ds.get('seed', row.get('seed','')))
                step = str(ds.get('step', row.get('step','')))
                degree = int(ds.get('degree', row.get('degree','')))
                if ds['benchmark']=='10D_nonproduct' and seed != '12345':
                    raise ValueError('Only seed 12345 is authorized for 10D non-product.')
                if '20D' in ds['benchmark'] and role in {'model','activation_control'}:
                    if seed not in {'12345','22345','32345'} or step != '25000':
                        raise ValueError('Unexpected 20D endpoint.')
                finite = math.isfinite(p)
                pc = min(1.0,max(0.0,p)) if finite else None
                raw_err = abs(p-ref)/ref*100 if finite else None
                clip_err = abs(pc-ref)/ref*100 if finite else None
                if finite:
                    assert 0 <= pc <= 1
                    assert clip_err <= raw_err+1e-10
                    old_err = row.get('relative_error_percent')
                    if not old_err and row.get('relative_error'):
                        old_err = 100*float(row['relative_error'])
                    if old_err and not math.isclose(float(old_err), raw_err, rel_tol=1e-7, abs_tol=1e-7):
                        warnings.append(dict(source=ds['file'], row=line, issue='source raw relative error differs from recomputation', source_error_percent=old_err, recomputed=raw_err))
                imag = row.get('imaginary_residual', row.get('raw_prediction_imaginary',''))
                if imag and (not math.isfinite(float(imag)) or abs(float(imag)) > 1e-12):
                    warnings.append(dict(source=ds['file'], row=line, issue='nonzero or nonfinite recorded imaginary residual', value=imag))
                points.append(dict(
                    benchmark=ds['benchmark'], dimension=ds['dimension'], method=method,
                    role=role, seed=seed, step=step, degree=degree, t=t, reference=ref,
                    reference_status=ds.get('reference_status','supplied_reference'),
                    raw_prediction=ptext, clipped_prediction=pc,
                    raw_relative_error_percent=raw_err, clipped_relative_error_percent=clip_err,
                    raw_out_of_bounds=int(finite and (p<0 or p>1)), nonfinite=int(not finite),
                    raw_invalid=int(not finite or p<0 or p>1), clipped_changed=int(finite and pc!=p),
                    imaginary_residual=imag,
                    source_invalid_flag=row.get('invalid_prediction',row.get('invalid_flag',row.get('invalid',row.get('invalid_raw_tail','')))),
                    source_error=row.get('evaluation_error',row.get('error_message',row.get('error',''))),
                    source_origin=ds['origin'], source_file=ds['file'], source_row=line,
                    source_sha256=origins[ds['file']]['sha256']))

    # A comparator, not a simulation reference or an NN4MGF evaluation.
    d10 = [r for r in points if r['benchmark']=='10D_nonproduct' and r['degree']==8]
    for row in d10:
        r = row.copy()
        p = -math.expm1(10*math.log1p(-math.exp(-r['t'])))
        err = abs(p-r['reference'])/r['reference']*100
        r.update(method='Independent_marginals',role='independence_comparator',seed='analytic',step='',degree=0,
                 raw_prediction=repr(p),clipped_prediction=p,raw_relative_error_percent=err,
                 clipped_relative_error_percent=err,raw_out_of_bounds=0,nonfinite=0,raw_invalid=0,clipped_changed=0,
                 imaginary_residual='',source_invalid_flag='',source_error='',
                 source_origin='analytic_comparator_with_pilot_reference',source_file='formula: 1-(1-exp(-t))^10',source_row='',source_sha256='')
        points.append(r)

    keys = ['benchmark','dimension','method','role','seed','step','degree','reference_status']
    groups = defaultdict(list)
    identities = set()
    for row in points:
        key = tuple(row[k] for k in keys)
        identity = key+(row['t'],)
        if identity in identities:
            raise ValueError(f'Duplicate point: {identity}')
        identities.add(identity)
        groups[key].append(row)
    per_run = []
    for key, rows in sorted(groups.items()):
        for hi in (5,8,10):
            if hi > max(r['t'] for r in rows):
                continue
            subset = [r for r in rows if 1 <= r['t'] <= hi]
            complete = {r['t'] for r in subset} == set(range(1,hi+1))
            nf = sum(r['nonfinite'] for r in subset)
            status = 'complete' if complete and nf==0 else 'unavailable'
            per_run.append(dict(zip(keys,key), interval=f'1:{hi}', n_points=len(subset), expected_points=hi,
                                status=status, raw_mre_percent=statistics.mean(r['raw_relative_error_percent'] for r in subset) if status=='complete' else None,
                                clipped_mre_percent=statistics.mean(r['clipped_relative_error_percent'] for r in subset) if status=='complete' else None,
                                raw_invalid_count=sum(r['raw_invalid'] for r in subset),
                                raw_out_of_bounds_count=sum(r['raw_out_of_bounds'] for r in subset), nonfinite_count=nf,
                                clipped_changed_count=sum(r['clipped_changed'] for r in subset)))
    aggkeys = [k for k in keys if k!='seed']+['interval']
    agroups = defaultdict(list)
    for row in per_run:
        agroups[tuple(row[k] for k in aggkeys)].append(row)
    aggregates = []
    for key, rows in sorted(agroups.items()):
        ok = all(r['status']=='complete' for r in rows)
        raw = [r['raw_mre_percent'] for r in rows]
        clipped = [r['clipped_mre_percent'] for r in rows]
        aggregates.append(dict(zip(aggkeys,key), seeds=';'.join(r['seed'] for r in rows), n_runs=len(rows),
                               status='complete' if ok else 'unavailable',
                               raw_mean_mre_percent=statistics.mean(raw) if ok else None,
                               clipped_mean_mre_percent=statistics.mean(clipped) if ok else None,
                               clipped_sample_sd_pp=statistics.stdev(clipped) if ok and len(rows)>1 else None,
                               raw_invalid_count=sum(r['raw_invalid_count'] for r in rows),
                               evaluated_point_count=sum(r['n_points'] for r in rows),
                               nonfinite_count=sum(r['nonfinite_count'] for r in rows)))

    # Same seed-12345 endpoint may appear in a trajectory table; it is not another run.
    for end in [r for r in points if r['benchmark']=='50D_product' and r['role']=='model']:
        matches = [r for r in points if r['benchmark']=='50D_product' and r['role']=='trajectory' and r['step']==end['step'] and r['degree']==end['degree'] and r['t']==end['t']]
        assert len(matches)==1
        assert float(matches[0]['raw_prediction'])==float(end['raw_prediction'])

    runtime = {}
    for method, updates in [('HCN',50000),('NN4MGF',300000)]:
        r = json.loads((root/f'inputs/runtime/{method}.json').read_text())
        assert r['mode']=='profile' and r['measured_updates']==200 and r['warmup_updates']==10
        projection = r['mean_update_seconds']*updates
        assert math.isclose(projection,r['projected_full_training_seconds_from_profile'],rel_tol=1e-12)
        runtime[method] = dict(r, planned_full_updates=updates, projected_full_hours=projection/3600,
                               planned_full_base_inputs=updates*r['base_inputs_per_update'],
                               peak_allocated_GiB=r['peak_allocated_bytes']/2**30,
                               peak_reserved_GiB=r['peak_reserved_bytes']/2**30,
                               peak_allocated_decimal_GB=r['peak_allocated_bytes']/1e9)
    h,n = runtime['HCN'],runtime['NN4MGF']
    runtime['ratios_NN4MGF_over_HCN'] = dict(mean_update_time=n['mean_update_seconds']/h['mean_update_seconds'],
                                           projected_full_time=n['projected_full_hours']/h['projected_full_hours'],
                                           peak_allocated=n['peak_allocated_bytes']/h['peak_allocated_bytes'])
    runtime['budget_conventions'] = dict(HCN=64000000, NN4MGF_paper_reported=300000*16384,
                                         NN4MGF_measured_code_path=300000*32768,
                                         HCN_percent_of_paper_reported=64000000/(300000*16384)*100,
                                         HCN_percent_of_profile_code=64000000/(300000*32768)*100)
    runtime['interpretation'] = '200 measured updates after 10 warmup updates; full-training hours are projections, not measured completed runs. Methods use different batches and parameter counts. No isolated attribution to removal of the CR penalty.'

    out = root/'results'
    out.mkdir(exist_ok=True)
    write_csv(out/'pointwise_clipped.csv', points)
    write_csv(out/'per_run_summary.csv', per_run)
    write_csv(out/'aggregate_summary.csv', aggregates)
    write_csv(out/'changed_predictions.csv', [r for r in points if r['clipped_changed']])
    # Keep signed inversion/network decomposition entirely raw.
    signed = []
    for r in points:
        if r['benchmark']=='20D_product' and r['role'] in {'model','activation_control','baseline'} and not r['nonfinite']:
            exact = [e for e in points if e['benchmark']==r['benchmark'] and e['role']=='inversion_diagnostic' and e['degree']==r['degree'] and e['t']==r['t']]
            assert len(exact)==1
            inv = float(exact[0]['raw_prediction'])
            p,ref = float(r['raw_prediction']),r['reference']
            a,b,c = (inv-ref)/ref*100,(p-inv)/ref*100,(p-ref)/ref*100
            assert math.isclose(a+b,c,rel_tol=1e-12,abs_tol=1e-8)
            signed.append(dict(benchmark=r['benchmark'],method=r['method'],seed=r['seed'],step=r['step'],degree=r['degree'],t=r['t'],reference=ref,raw_exact_inversion=inv,raw_prediction=p,raw_signed_inversion_error_percent=a,raw_signed_network_error_percent=b,raw_signed_total_error_percent=c))
    write_csv(out/'signed_error_components_raw.csv', signed)
    (out/'runtime_summary.json').write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps(dict(source_hashes_verified=len(manifest['files']),
        point_count=len(points), finite_point_count=sum(not r['nonfinite'] for r in points),
        duplicate_points=0, selected_10d_seeds=sorted({r['seed'] for r in points if r['benchmark']=='10D_nonproduct' and r['role']=='model'}),
        product_references_verified=True, clipping_projection_checks_passed=True,
        trajectory_endpoint_crosscheck_passed=True, warnings=warnings),ensure_ascii=False,indent=2)+'\n')

    def get(bench,method,degree,interval,role=None):
        rows = [r for r in aggregates if r['benchmark']==bench and r['method']==method and r['degree']==degree and r['interval']==interval and (role is None or r['role']==role)]
        assert len(rows)==1,(bench,method,degree,interval)
        return rows[0]
    def fmt(x):
        return '—' if x is None else f'{x:.3f}'
    lines = [
        '# HCN 实验代码与 clipped 结果核对（2026-10-03）', '',
        '本次整理以新上传的 `formal experiment.zip` 为主。按作者补充要求，30D/50D product-form 使用先前 draft 对应的固定终点逐点 CSV；这些 CSV 的 raw MRE 与旧稿表 6、7 一致。所有计算使用 CSV 的完整精度，不从旧稿已舍入的 MRE 反推。', '',
        '## 1. 采用的规则', '',
        '- 所有有限实数尾概率统一投影到 `[0,1]`，随后逐点重算相对误差和区间 MRE；保留原值、原始越界标记及来源。',
        '- NaN/Inf 不转成 0 或 1。若某报告区间含非有限点，该区间 MRE 标为不可用，不通过删除失败点得到均值。本次纳入的逐点记录全部有限。',
        '- 10D non-product 只纳入 seed 12345 的 step 25000；不汇总另两个 seed。它是单次 pilot 展示，不提供跨 seed 稳健性证据。30D non-product 未纳入，也未检查其内容。',
        '- 20D 按三个 seed 的固定 step 25000 分别算 MRE，再报告均值及样本标准差。三个重复评估目录中的对应 CSV 完全相同，只计一次。',
        '- 保留所有已有 degree 的结果，主规则维持 2D Degree 5、高维 Degree 8；不根据本次重算选择更好的 degree 或 checkpoint。', '',
        '公式：`MRE (%) = 100 × mean(|clip(p_raw,0,1) − p_ref| / p_ref)`。表内 SD 单位为百分点，采用 `ddof=1`；单次运行不报告 SD。', '',
        '## 2. 主要结果', '',
        '| 实验/方法 | Degree | 运行数 | MRE 1:5 (%) | MRE 1:8 (%) | MRE 1:10 (%) |',
        '|---|---:|---:|---:|---:|---:|']
    table_specs = [
        ('2D_product','HCN',5,'2D product / HCN'),
        ('2D_product','NN4MGF_published',5,'2D product / NN4MGF published'),
        ('2D_Harrison','HCN',5,'2D Harrison / HCN'),
        ('2D_Harrison','NN4MGF_published',5,'2D Harrison / NN4MGF published'),
        ('10D_nonproduct','HCN_interaction',8,'10D non-product / HCN（pilot）'),
        ('10D_nonproduct','Independent_marginals',0,'10D / independent-marginal comparator'),
        ('20D_product','HCN_Cauchy',8,'20D product / HCN，三 seed 均值'),
        ('20D_product','NN4MGF_archived',8,'20D product / NN4MGF archived'),
        ('20D_product','Complex_tanh',8,'20D product / complex tanh，三 seed 均值'),
        ('30D_product','HCN',5,'30D product / HCN，同 degree 比较'),
        ('30D_product','NN4MGF_published',5,'30D product / NN4MGF published'),
        ('30D_product','HCN',8,'30D product / HCN，主结果'),
        ('50D_product','HCN',8,'50D product / HCN，主结果')]
    for bench,method,degree,label in table_specs:
        chosen = [r for r in aggregates if r['benchmark']==bench and r['method']==method and r['degree']==degree and r['role']!='trajectory']
        vals = {r['interval']:r['clipped_mean_mre_percent'] for r in chosen}
        lines.append(f'| {label} | {degree if degree else "—"} | {chosen[0]["n_runs"]} | {fmt(vals.get("1:5"))} | {fmt(vals.get("1:8"))} | {fmt(vals.get("1:10"))} |')
    h20 = get('20D_product','HCN_Cauchy',8,'1:8')
    lines += ['',f'20D HCN Degree 8 的 1:8 结果为 **{h20["clipped_mean_mre_percent"]:.3f}% ± {h20["clipped_sample_sd_pp"]:.3f} 个百分点**（三个 seed 的均值 ± 样本 SD）。这里使用25k固定终点，不混入旧探索中的精修、五 seed 或最佳 checkpoint 数字。', '',
              '10D 的参考数组为 `[0.97204, 0.72091, 0.40030, 0.19124, 0.08443, 0.03576, 0.014645, 0.005905]`。文件将其标为未经认证的 screenshot pilot，无经验证的离散误差界；因此这里只能称为相对 pilot reference 的误差。', '',
              '2D/30D NN4MGF 使用已发布表中的有限精度数值；20D 使用归档的 released-checkpoint 全精度评估，未重新运行 NN4MGF，且此处不声称相同训练预算。', '',
              '## 3. Clipping 对重点结果的影响', '',
              '| 实验 | Degree | 阈值 | raw | clipped |', '|---|---:|---:|---:|---:|']
    for r in points:
        if r['role']=='model' and r['degree']==8 and r['benchmark'] in {'10D_nonproduct','30D_product','50D_product'} and r['clipped_changed']:
            lines.append(f'| {r["benchmark"]} | 8 | {r["t"]:g} | {float(r["raw_prediction"]):.12g} | {r["clipped_prediction"]:g} |')
    lines += ['', '| 实验 | raw MRE 1:8 (%) | clipped MRE 1:8 (%) | 原始越界点 / 8 |', '|---|---:|---:|---:|']
    for bench,method in [('10D_nonproduct','HCN_interaction'),('30D_product','HCN'),('50D_product','HCN')]:
        r = get(bench,method,8,'1:8','model')
        lines.append(f'| {bench} | {r["raw_mean_mre_percent"]:.6f} | {r["clipped_mean_mre_percent"]:.6f} | {r["raw_invalid_count"]}/8 |')
    lines += ['', '30D/50D 主结果都有轻微 overshoot，而非完全没有越界。Clipping 对它们的整体 MRE 影响很小。10D 的主要变化来自 t=1 的大幅越界；clipping 将报告值投影回合法区间，但不代表修复了产生该 raw 输出的数值机制。', '',
              '20D complex-tanh 的 seed 22345 / Degree 8 原始 MRE 非常大，clipping 后可成为有限的区间结果。按统一规则应保留该 seed 并重算三 seed 汇总；旧稿中“只对两个成功 tanh seed 求均值”的叙述需要更新。原始不稳定性由 raw 越界和原值单独呈现。', '',
              '## 4. 代码与 Section 3 的对应', '',
              f'本次对新代码与 Section 3 写作所用代码进行了 {len(manifest["code_ast_checks"])} 项函数/类 AST 比对，所查单元一致，包括 Cauchy 激活、坐标共享/全局分支、BAR 和边界/实轴结构损失。没有发现需要改动 Section 3 核心网络或损失公式的新差异。该比对不等同于对全部实现或理论结论的形式化证明。', '',
              '- 原点归一化和边界质量由结构硬约束实现；没有另加原点拟合损失。',
              '- Cauchy 全纯性限定在无极点区域；去除 CR penalty 后，参数优化仍需自动微分。',
              '- 10D 的 global interaction 分支进入内部与边界网络；边界分支屏蔽本面的坐标。2D Harrison 的全局扩展进入内部网络，不能把二者写成完全相同结构。',
              '- 10D 采样实部下界为 −0.2，2D/product-form 家族为 −0.5；训练细节需要区别写明。',
              '- 2D 每次更新 3416+427+427 个基础输入，15000 次更新合计 **64.05M**。10D、20D、30D、50D 对应固定终点合计 **64M**。',
              '- 30D/50D 的历史结果绑定各自固定终点，不使用 smoke 输出代替正式结果。此次已将旧稿的逐点结果和 checkpoint 身份核对补齐。', '',
              '## 5. 同卡 runtime 与显存', '',
              '两份报告均来自 NVIDIA H100 NVL：先 warmup 10 次，再测量 200 次更新。', '',
              '| 项目 | HCN | NN4MGF |', '|---|---:|---:|',
              f'| 平均每次更新（秒） | {h["mean_update_seconds"]:.6f} | {n["mean_update_seconds"]:.6f} |',
              f'| 200 次更新测量时间（秒） | {h["measured_update_seconds"]:.3f} | {n["measured_update_seconds"]:.3f} |',
              f'| 峰值 allocated（GiB） | {h["peak_allocated_GiB"]:.3f} | {n["peak_allocated_GiB"]:.3f} |',
              f'| 峰值 reserved（GiB） | {h["peak_reserved_GiB"]:.3f} | {n["peak_reserved_GiB"]:.3f} |',
              f'| 每次更新基础输入数 | {h["base_inputs_per_update"]} | {n["base_inputs_per_update"]} |',
              f'| 对应完整更新数 | 50,000 | 300,000 |',
              f'| 按 profile 外推完整时间（小时） | {h["projected_full_hours"]:.3f} | {n["projected_full_hours"]:.3f} |', '',
              f'完整 schedule 的时间比约为 **{runtime["ratios_NN4MGF_over_HCN"]["projected_full_time"]:.2f}×（外推）**，allocated 显存比约为 **{runtime["ratios_NN4MGF_over_HCN"]["peak_allocated"]:.2f}×**。不能称为完整训练实测加速比，不能写成等 batch/等参数量的吞吐比较，也不能把全部差异归因于去除 CR penalty。', '',
              '这里实际计时代码的 NN4MGF 每次 optimizer update 有两次 batch=16384 的采样/反向传播，基础输入为32768。按该代码路径的300k更新，预算为9.8304B，HCN的64M占约0.6510%。按旧稿引用的论文口径300k×16384，则为4.9152B，占约1.3021%。正文必须明确区分“论文报告预算”和“本次计时代码路径预算”，不能混写成同一个数。', '',
              '## 6. 后续正文更新时的注意项', '',
              '- 评估协议、图注、表注统一改为 clipped probability；附表保留 raw invalid count 和 raw 原值。不要只替换表内 MRE 而保留“without clipping”。',
              '- 20D 激活消融按所有三个有限运行重算；不把 clipping 后的有限预测仍按旧 raw 成功规则剔除。',
              '- exact-transform 数值反演诊断也同时提供 raw/clipped 汇总。解释数值机制时保留 raw 误差；有符号的 inversion/network/total 加法分解不可混用 clipped total 和 raw 分量。本包额外保留全 raw 分解表。',
              '- 50D late-training 表使用同一条 seed-12345 轨迹，不算六个独立 seed；100k 固定终点在 trajectory 中再次出现，仅作为对照，不重复计入主结果。',
              '- 本次没有训练或加载 checkpoint 做推理。已验证9份 checkpoint 文件的 SHA-256 身份；这只能确认文件一致，不能代替独立复现推理。', '',
              '## 7. 数据与复算', '',
              '`inputs/` 保存本次采用的原始 CSV/JSON，`selection_manifest.json` 记录原路径、SHA-256、去重关系、checkpoint 校验和代码比对。', '',
              '`results/pointwise_clipped.csv`：逐点 raw/clipped、误差、状态与来源。', '',
              '`results/per_run_summary.csv`：按 seed、终点、degree、区间汇总。', '',
              '`results/aggregate_summary.csv`：跨 seed 均值与样本 SD。', '',
              '`results/changed_predictions.csv`：全部被 clipping 改动的点。', '',
              '`results/signed_error_components_raw.csv`：20D 全 raw 有符号误差分解。', '',
              '`results/runtime_summary.json` 与 `results/validation.json`：资源统计及核对记录。', '',
              '复算：解压完整数据包，在文件夹内执行 `python audit_results.py`。只需 Python 标准库；不需要 GPU、PyTorch 或重新训练。', '']
    (root/'HCN_Formal_Experiment_Audit_20261003.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(points=len(points),per_run_summaries=len(per_run),aggregate_summaries=len(aggregates),warnings=len(warnings),output=str(out)),ensure_ascii=False))


if __name__ == '__main__':
    main()
