"""Rebuild paper-only summaries from immutable recorded predictions; no training."""
from pathlib import Path
import csv
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/d20_activation_ablation'

def write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    with (DATA / 'pointwise.csv').open(newline='') as f:
        raw = list(csv.DictReader(f))
    # These are the manuscript's explicitly reported outcome classifications,
    # not a newly fitted threshold or automatic exclusion of invalid points.
    shared = (12345, 32345)
    rows = []
    for method in ('Cauchy', 'tanh'):
        for end in (5, 8, 10):
            means = []
            for seed in shared:
                points = [r for r in raw if r['method'] == method and int(r['seed']) == seed
                          and int(r['degree']) == 8 and 1 <= int(r['t']) <= end]
                assert len(points) == end
                assert {int(r['t']) for r in points} == set(range(1, end + 1))
                means.append(statistics.mean(100 * abs(float(r['raw_prediction']) - float(r['true_value']))
                                             / float(r['true_value']) for r in points))
            rows.append(dict(method=method, degree=8, interval=f'1:{end}',
                             seeds='12345;32345', n_seeds=2, mre_percent=statistics.mean(means)))
    write(DATA / 'paired_success_summary.csv', rows)
    write(DATA / 'failure_summary.csv', [
        dict(method='Cauchy', attempted_runs=3, usable_runs=3, failed_seeds='',
             classification='all three recorded runs usable'),
        dict(method='tanh', attempted_runs=3, usable_runs=2, failed_seeds='22345',
             classification='manuscript-reported catastrophic numerical failure; raw values retained')])
    print('Wrote paired summaries and failure counts; all-seed summaries and raw predictions unchanged.')

if __name__ == '__main__':
    main()
