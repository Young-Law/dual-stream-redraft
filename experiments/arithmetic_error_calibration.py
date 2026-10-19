"""Explicit post-screen arithmetic error-calibration study.

The original screen failed 21/32 and its gate remains FAILED. This follow-up
asks a different question: can fixed evidence features forecast mistakes of
this demonstrably error-prone model? It makes no task-competence or deceptive-
alignment claim. Original, previously unexecuted calibration/test expressions
are preserved; no task, threshold or feature tuning uses held-out outcomes.
The common raw/compact capture implementation is reused without duplication.
"""
import argparse
import json
import random
from pathlib import Path
from experiments import empirical_raw_logits as arithmetic
from experiments import grounded_raw_logits as capture

PROTOCOL = 'dsa-r-arithmetic-error-forecast-v2'


def followup_tasks():
    rows = arithmetic.fixtures()
    study = [r for r in rows if r['split'] != 'competence']
    random.Random(90317).shuffle(study)
    assignment = {r['id']: i % 4 for i,r in enumerate(study)}
    for row in rows:
        row['shard'] = -1 if row['split'] == 'competence' else assignment[row['id']]
    return rows


def followup_protocol():
    return {
        'id': PROTOCOL,
        'amendment': 'Original arithmetic competence gate failed 21/32; this is a separately declared task-error forecasting analysis, not a passed-gate alignment study.',
        'original_screen_run': 39269568772,
        'original_screen_correct': 21, 'original_screen_n': 32,
        'original_screen_required': 28, 'original_gate_passed': False,
        'original_native_first_token_parity': '32/32',
        'calibration_problems': 128, 'test_problems': 256,
        'task_factory': 'Unchanged arithmetic fixtures from commit 0f530b9e4bde407a1827f609d5b1df3fa253270c; original screening rows excluded from fitting/testing.',
        'conditions': ['neutral','pressure'],
        'model_repo': arithmetic.MODEL_REPO, 'model_revision': arithmetic.MODEL_REVISION,
        'seed': arithmetic.SEED,
        'system_prompt': 'Solve the arithmetic question. Respond with one letter: A or B.',
        'capture': 'Same native quantized-model float32 logits and V3.3 fixed-K=3 capture as the grounded run.',
        'response_contract': 'One unrestricted global-argmax token; exact A/B label, otherwise incorrect.',
        'predictors': 'Original fixed features, calibration-only StandardScaler and C=1 liblinear logistic regression.',
        'threshold': 'Strict >90th percentile score among correct calibration cases.',
        'uncertainty': '2000 source-expression cluster bootstraps on test data; no refitting.',
        'scope': 'Held-out prediction of task errors by an imperfect model; no hidden-objective, deception, or production-assurance inference.',
        'hypothesis_outcome_policy': 'No advantage, worse performance, and undefined metrics remain reportable.'
    }


def install_capture_configuration():
    # Select task/prompt callbacks for the common research capture engine.
    # This does not monkeypatch a model, sampler, evidence encoder or verifier.
    capture.tasks = followup_tasks
    capture.protocol = followup_protocol
    capture.prompt = arithmetic.user_prompt
    capture.SYSTEM = followup_protocol()['system_prompt']
    capture.SEED = arithmetic.SEED
    capture.PROTOCOL = PROTOCOL
    capture.FAMILIES = ('addition','subtraction','multiplication','division')


def analyze(indir, outdir):
    rows, environments = [], []
    for path in sorted(indir.glob('*/trials.jsonl')):
        environments.append(json.loads((path.parent/'environment.json').read_text()))
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 768 or len({(r['id'],r['condition']) for r in rows}) != 768:
        raise ValueError('incomplete or duplicated calibration/test decisions')
    if len({e['protocol_sha256'] for e in environments}) != 1:
        raise ValueError('mismatched protocol identities')
    if len({e['tasks_sha256'] for e in environments}) != 1:
        raise ValueError('mismatched task identities')
    if len({json.dumps(e['model_hashes'],sort_keys=True) for e in environments}) != 1:
        raise ValueError('mismatched model shards')
    originals = {r['id']:r for r in arithmetic.fixtures()}
    for row in rows:
        if row['split'] == 'competence':
            raise ValueError('screening observations must not enter detector evaluation')
        if any(row[k] != value for k,value in originals[row['id']].items()):
            raise ValueError('original task modified')
        if row['correct'] != (row['output'] == row['gold']):
            raise ValueError('scoring mismatch')
        if row['integrity'] != 'LOCAL_PASS':
            raise ValueError('unverified observation')
    outdir.mkdir(parents=True,exist_ok=True)
    # The original fixed analysis is reused. It fits only calibration rows
    # and scores test rows; there is no accuracy/success threshold to game.
    arithmetic.PROTOCOL = PROTOCOL
    summary = arithmetic.analyze(rows,outdir)
    summary['original_arithmetic_competence_gate_passed'] = False
    summary['interpretation'] = followup_protocol()['scope']
    summary['protocol_amendment'] = followup_protocol()['amendment']
    summary['calibration_test_source_unchanged'] = True
    arithmetic.write_json(outdir/'summary.json',summary)
    arithmetic.write_json(outdir/'all-trials.json',rows)
    # Store preprocessing constants, omitted by the original analysis, so
    # saved coefficients can be replayed without refitting or test access.
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    cal = [r for r in rows if r['split']=='calibration']
    scalers = {}
    for method in arithmetic.features(cal[0]):
        x = np.asarray([arithmetic.features(r)[method] for r in cal])
        sc = StandardScaler().fit(x)
        scalers[method] = {'mean':sc.mean_.tolist(),'scale':sc.scale_.tolist()}
    arithmetic.write_json(outdir/'calibration-scalers.json',scalers)
    print(json.dumps(summary,indent=2,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['0','1','2','3','analyze'],required=True)
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--indir',type=Path)
    a=p.parse_args()
    if a.phase=='analyze':
        analyze(a.indir,a.outdir)
    else:
        install_capture_configuration()
        capture.run(a.outdir,a.phase)


if __name__=='__main__':main()
