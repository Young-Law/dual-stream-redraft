"""Secondary precision comparison specified before grounded test outcomes.

The direction is fixed: higher 1-P(top-one) predicts error. Precision-specific
thresholds use correct calibration cases only; no test fitting is performed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def compute(rows, resamples=2000):
    cal = sorted([r for r in rows if r['split'] == 'calibration'], key=lambda r:(r['id'], r['condition']))
    test = sorted([r for r in rows if r['split'] == 'test'], key=lambda r:(r['id'], r['condition']))
    if len(cal) != 256 or len(test) != 512:
        raise ValueError('incomplete calibration or test data')
    yc = np.array([not r['correct'] for r in cal])
    yt = np.array([not r['correct'] for r in test])

    def scores(rs):
        p = np.array([r['top1_probability'] for r in rs], dtype=np.float64)
        return {
            'float64_from_raw': 1-p,
            'float32_probability': 1-p.astype(np.float32).astype(float),
            'float16_probability': 1-p.astype(np.float16).astype(float),
            'retained_uint8': 1-np.array([r['compact_top1_probability'] for r in rs]),
        }

    def auc(y, s):
        return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else None

    cs, ts = scores(cal), scores(test)
    stat = {}
    for method, x in ts.items():
        threshold = float(np.quantile(cs[method][~yc], .9, method='higher')) if (~yc).any() else None
        det = x > threshold if threshold is not None else np.zeros_like(yt)
        stat[method] = {
            'test_auc': auc(yt, x),
            'test_rounded_probability_one': int((x == 0).sum()),
            'test_unique_score_count': len(set(x.tolist())),
            'calibration_threshold': threshold,
            'test_false_positive_count': int(det[~yt].sum()),
            'test_correct_n': int((~yt).sum()),
            'test_true_positive_count': int(det[yt].sum()),
            'test_error_n': int(yt.sum()),
            'test_fpr': float(det[~yt].mean()) if (~yt).any() else None,
            'test_recall': float(det[yt].mean()) if yt.any() else None,
        }
    groups = sorted({r['id'] for r in test})
    indices = {g:[j for j,r in enumerate(test) if r['id'] == g] for g in groups}
    rng = np.random.default_rng(99401)
    boot = {m:[] for m in ts}
    diffs = []
    for _ in range(resamples):
        selected = rng.choice(groups, len(groups), replace=True)
        ix = np.array([j for g in selected for j in indices[g]])
        if len(set(yt[ix].tolist())) != 2:
            continue
        values = {m:auc(yt[ix], x[ix]) for m,x in ts.items()}
        for m,v in values.items():
            boot[m].append(v)
        diffs.append(values['float64_from_raw'] - values['retained_uint8'])

    def ci(values):
        return np.quantile(values, [.025, .975]).tolist() if values else None

    for m in stat:
        stat[m]['test_auc_cluster_bootstrap_95pct_ci'] = ci(boot[m])
    return {
        'scope': 'Secondary scalar precision comparison; specified before grounded held-out outcomes; not a safety guarantee',
        'source_protocol': 'dsa-r-grounded-records-v1',
        'score': '1 - P(pre-control top-one)',
        'calibration_n': len(cal), 'test_n': len(test),
        'test_problem_clusters': len(groups),
        'bootstrap_resamples': resamples, 'bootstrap_seed': 99401,
        'test_labels_used_for_thresholds': False,
        'methods': stat,
        'raw_minus_uint8_auc_95pct_ci': ci(diffs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('trials', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = json.loads(a.trials.read_text(encoding='utf-8'))
    result = compute(rows)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
