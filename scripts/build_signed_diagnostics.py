from pathlib import Path
import csv, math
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results'
def read(name):
    with (DATA/name).open(newline='') as f: return list(csv.DictReader(f))
exact={(int(x['degree']),float(x['t'])):x for x in read('d20_skew_symmetric_product_form/exact_transform_diagnostic.csv')}
models=read('d20_skew_symmetric_product_form/pointwise.csv')+list(csv.DictReader((ROOT/'experiments/d20_activation_ablation/reference/nn4mgf_archived_pointwise.csv').open(newline='')))
rows=[]
for x in models:
    degree=int(x['degree']);t=float(x['t'])
    if degree not in (5,8):continue
    ref=float(x['true_value']);ex=exact[(degree,t)]
    assert math.isclose(ref,float(ex['true_value']),rel_tol=1e-12)
    star=float(ex['raw_prediction']);pred=float(x['raw_prediction'])
    inv=100*(star-ref)/ref;net=100*(pred-star)/ref;total=100*(pred-ref)/ref
    assert math.isclose(inv+net,total,rel_tol=1e-12,abs_tol=1e-12)
    rows.append(dict(model='HCN' if x['method']=='Cauchy' else 'NN4MGF',seed=x['seed'],degree=degree,t=int(t),reference=ref,exact_inversion=star,prediction=pred,inversion_signed_percent=inv,replacement_signed_percent=net,total_signed_percent=total))
rows.sort(key=lambda x:(x['degree'],x['t'],x['model'],x['seed']))
assert len(rows)==80
out=DATA/'diagnostics';out.mkdir(exist_ok=True)
with (out/'d20_signed_errors.csv').open('w',newline='') as f:
    dw=csv.DictWriter(f,fieldnames=list(rows[0]));dw.writeheader();dw.writerows(rows)
print('Verified and wrote 80 signed decompositions. No training or checkpoint inference performed.')
