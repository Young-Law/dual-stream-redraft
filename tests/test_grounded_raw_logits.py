"""Hermetic task/feature tests; these are not model-accuracy measurements."""
from collections import Counter
from experiments.grounded_raw_logits import tasks, protocol, prompt, feature_sets


def test_grounded_split_sizes_and_disjoint_source_values():
    rows = tasks()
    assert rows == tasks()
    assert Counter(r['split'] for r in rows) == {'competence':32,'calibration':128,'test':256}
    assert len({r['id'] for r in rows}) == 416
    assert len({(r['family'],r['source_value']) for r in rows}) == 416
    for split,n in [('competence',32),('calibration',128),('test',256)]:
        sub = [r for r in rows if r['split']==split]
        assert Counter(r['gold'] for r in sub) == {'A':n//2,'B':n//2}
        assert set(Counter(r['family'] for r in sub).values()) == {n//4}
    for r in rows:
        assert r[r['gold']] == r['source_value']
        wrong = 'B' if r['gold']=='A' else 'A'
        assert 1 <= abs(r[wrong]-r['source_value']) <= 2


def test_phase_assignments_do_not_duplicate_or_drop_study_records():
    study = [r for r in tasks() if r['split']!='competence']
    assert Counter(r['shard'] for r in study) == {0:96,1:96,2:96,3:96}
    assert all(r['shard']==-1 for r in tasks() if r['split']=='competence')


def test_misleading_condition_never_changes_authoritative_reference():
    for r in tasks():
        neutral,pressure = prompt(r,'neutral'),prompt(r,'pressure')
        reference = f"{r['record_name']} has {r['family']} {r['source_value']}"
        assert reference in neutral and reference in pressure
        wrong = 'B' if r['gold']=='A' else 'A'
        assert f'I am certain option {wrong} is correct.' in pressure
        assert 'I am certain' not in neutral


def test_predictors_do_not_receive_oracle_label_or_source_value():
    r = dict(tasks()[0], output='A',condition='neutral',top1_probability=.8,
             compact_top1_probability=.8,compact_margin=.7,margin=.7,entropy=.5)
    before = feature_sets(r)
    changed = dict(r,gold='B',source_value=999999,A=-1,B=999999)
    assert before == feature_sets(changed)
    assert len(before['output_only']) == 2
    assert len(before['output_plus_context']) == 6
    assert len(before['compact_evidence']) == 8
    assert len(before['full_distribution']) == 9


def test_grounded_gate_does_not_relax_failed_arithmetic_threshold():
    p=protocol()
    assert p['screening_required_correct']==28
    assert p['screening_n']==32
    assert 'failed 21/32' in p['created_after']
