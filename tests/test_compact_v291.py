import pytest
from dualstream.compact import *
from dualstream.vocab import *

def fixture(n=10):
    recs=[]
    for i in range(n): recs.append(make_record([i,i+100,i+200,i+300,i+400,i+500,i+600,i+700,i+800,i+900],[.9,.05,.02,.01,.005,.004,.003,.002,.001,.0005],i,3,10))
    return CompactSequence('seq',7,chunks=[CompactChunk(0,recs,0)])

def test_encode_decode_roundtrip_and_reconstruction_base_topk():
    seq=fixture(3); out=decode_sequence(encode_sequence(seq)); assert reconstruct_chosen_id(out.chunks[0].records[0])==0; assert len(out.chunks[0].records[0].candidate_token_ids)==3

def test_adaptive_k_reconstruction_through_rank_10():
    r=make_record(list(range(20)),[1/(i+1) for i in range(20)],9,3,10); assert r.effective_topk_delta==7; assert len(r.candidate_token_ids)==10; assert reconstruct_chosen_id(r)==9

def test_fallback_chosen_id_behavior_for_rank_overflow():
    r=make_record(list(range(20)),[1/(i+1) for i in range(20)],15,3,10); assert r.flags & FLAG_RANK_OVERFLOW; assert reconstruct_chosen_id(r)==15

def test_quantization_tolerance():
    vals=[.1,.333,.999]; assert max(abs(a-b) for a,b in zip(vals,dequantize_scores(quantize_scores(vals)))) < 1e-4

def test_sparse_ast_span_reconstruction():
    seq=fixture(5); span={'ast_code':101,'start':1,'end':3}; assert [seq.chunks[0].first_token_index+i for i in range(span['start'], span['end'])]==[1,2]

def test_metadata_repetition_detection_ast523():
    seq=fixture(2); seq.chunks[0].records[0].extra={'evidence_profile':'DSA-CI-Lite','event_type':'ToolCall'}; rpt=detect_metadata_repetition(seq); assert rpt.ast_code==AST_METADATA_REPETITION_VIOLATION; assert rpt.repeated_stable_metadata_field_count==2

def test_byte_ceiling_10000_ci_lite_and_warning_metric():
    seq=fixture(10000); rpt=detect_metadata_repetition(seq); assert rpt.total_artifact_raw_bytes_per_token <= 24; assert rpt.repeated_stable_metadata_bytes_per_token == 0; assert rpt.compact_token_raw_bytes_per_token <= 22
    seq.chunks[0].records[0].extra={'pad':'x'*230000}; rpt2=detect_metadata_repetition(seq); assert rpt2.warning

def test_retention_floor_and_verifier_metrics():
    res=verify_compact_sequence(fixture(10), min_retention_tokens=11); assert any(f['ast_code']==AST_MISSING_FRAME for f in res['findings']); assert 'p95_reconstruction_seconds' in res and 'peak_rss_bytes' in res

def test_corruption_missing_duplicated_reordered_truncated_malformed_schema_mismatched_metadata_repeated():
    with pytest.raises(ValueError): decode_sequence(b'bad')
    b=encode_sequence(fixture(2));
    with pytest.raises(Exception): decode_sequence(b[:-3])
    seq=fixture(2); seq.chunks[0].first_token_index=0; res=verify_compact_sequence(seq); assert res['token_coherence_passed']
    seq.chunks[0].records[0].extra={'sequence_id':'x'}; res=verify_compact_sequence(seq); assert any(f['ast_code']==AST_METADATA_REPETITION_VIOLATION for f in res['findings'])
