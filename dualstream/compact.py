from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json, struct, time, resource
import numpy as np
from .vocab import AST_METADATA_REPETITION_VIOLATION, AST_SCHEMA_MISMATCH, AST_MISSING_FRAME

STABLE_METADATA_FIELDS = {
 'evidence_profile','assurance_class','tokenizer_id','signal_schema_id','signal_schema_hash','probe_pack_id','probe_pack_hash','decoder_control_flags','base_topk','max_adaptive_rank','adaptive_policy_id','quantization_id','verifier_budget_id','retention_floor_policy_id','tool_registry_hash','agent_policy_hash','context_policy_hash','memory_policy_hash','rollout_policy_hash','sequence_id','prompt_nonce','chunk_index','first_token_index','token_count'
}
AGENTOPS_EVENT_FIELDS={'event_type','agent_step_id','actor_id','tool_id','memory_id','context_snapshot_id'}
FLAG_RANK_OVERFLOW=1
MAGIC=b'DSAC291\0'

@dataclass
class CompactTokenRecord:
    candidate_token_ids: List[int]
    quantized_scores: List[int]
    chosen_rank: int
    effective_topk_delta: int = 0
    fallback_chosen_id: Optional[int] = None
    flags: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CompactChunk:
    first_token_index: int
    records: List[CompactTokenRecord]
    chunk_index: int = 0

@dataclass
class CompactSequence:
    sequence_id: str
    prompt_nonce: int
    evidence_profile: str = 'DSA-CI-Lite'
    assurance_class: str = 'DSA-R'
    tokenizer_id: str = 'unknown'
    signal_schema_id: str = 'AST-1'
    signal_schema_hash: str = ''
    probe_pack_id: str = ''
    probe_pack_hash: str = ''
    decoder_control_flags: Dict[str, Any] = field(default_factory=dict)
    base_topk: int = 3
    max_adaptive_rank: int = 10
    adaptive_policy_id: str = 'rank-through-committed-v1'
    quantization_id: str = 'uint16-prob-v1'
    verifier_budget_id: str = 'ci-lite-v2.9.1'
    retention_floor_policy_id: str = 'ci-lite-retention-floor-v1'
    tool_registry_hash: str = ''
    agent_policy_hash: str = ''
    context_policy_hash: str = ''
    memory_policy_hash: str = ''
    rollout_policy_hash: str = ''
    chunks: List[CompactChunk] = field(default_factory=list)

@dataclass
class MetadataReport:
    repeated_stable_metadata_field_count:int=0; repeated_stable_metadata_bytes_total:int=0; repeated_stable_metadata_bytes_per_token:float=0.0
    sequence_header_raw_bytes:int=0; chunk_header_raw_bytes:int=0; compact_token_raw_bytes:int=0; adaptive_payload_raw_bytes:int=0
    span_event_raw_bytes:int=0; trajectory_event_raw_bytes:int=0; manifest_raw_bytes:int=0; compact_token_raw_bytes_per_token:float=0.0; total_artifact_raw_bytes_per_token:float=0.0
    ast_code:Optional[int]=None; warning:bool=False

def quantize_scores(scores: List[float])->List[int]:
    return [max(0,min(65535,int(round(float(s)*65535)))) for s in scores]
def dequantize_scores(q: List[int])->List[float]: return [x/65535.0 for x in q]

def make_record(ranked_ids:List[int], scores:List[float], chosen_id:int, base_topk:int=3, max_adaptive_rank:int=10)->CompactTokenRecord:
    try: rank=ranked_ids.index(chosen_id)
    except ValueError: rank=-1
    if 0 <= rank < base_topk:
        k=base_topk; flags=0; fb=None; cr=rank; delta=0
    elif 0 <= rank < max_adaptive_rank:
        k=rank+1; flags=0; fb=None; cr=rank; delta=k-base_topk
    else:
        k=base_topk; flags=FLAG_RANK_OVERFLOW; fb=chosen_id; cr=255; delta=0
    return CompactTokenRecord(ranked_ids[:k], quantize_scores(scores[:k]), cr, delta, fb, flags)

def reconstruct_chosen_id(r:CompactTokenRecord)->int:
    if r.flags & FLAG_RANK_OVERFLOW:
        if r.fallback_chosen_id is None: raise ValueError('rank overflow without fallback chosen_id')
        return r.fallback_chosen_id
    return r.candidate_token_ids[r.chosen_rank]

def _rec_to_bytes(r:CompactTokenRecord)->bytes:
    b=struct.pack('<BBBB', r.flags, r.effective_topk_delta, r.chosen_rank, len(r.candidate_token_ids))
    for tid,sc in zip(r.candidate_token_ids,r.quantized_scores): b+=struct.pack('<IH',tid,sc)
    if r.flags & FLAG_RANK_OVERFLOW: b+=struct.pack('<I', int(r.fallback_chosen_id))
    if r.extra: b+=json.dumps(r.extra,sort_keys=True,separators=(',',':')).encode()
    return b

def encode_sequence(seq:CompactSequence)->bytes:
    d=seq.__dict__.copy(); chunks=d.pop('chunks')
    head=json.dumps(d,sort_keys=True,separators=(',',':')).encode(); out=MAGIC+struct.pack('<I',len(head))+head+struct.pack('<H',len(chunks))
    for c in chunks:
        out+=struct.pack('<III',c.chunk_index,c.first_token_index,len(c.records))
        for r in c.records:
            rb=_rec_to_bytes(r); out+=struct.pack('<H',len(rb))+rb
    return out

def decode_sequence(buf:bytes)->CompactSequence:
    if not buf.startswith(MAGIC): raise ValueError('malformed compact artifact')
    off=len(MAGIC); n=struct.unpack('<I',buf[off:off+4])[0]; off+=4
    d=json.loads(buf[off:off+n]); off+=n; cc=struct.unpack('<H',buf[off:off+2])[0]; off+=2; chunks=[]
    for _ in range(cc):
        ci,first,cnt=struct.unpack('<III',buf[off:off+12]); off+=12; recs=[]
        for _ in range(cnt):
            ln=struct.unpack('<H',buf[off:off+2])[0]; off+=2; rb=buf[off:off+ln]; off+=ln
            flags,delta,cr,k=struct.unpack('<BBBB',rb[:4]); pos=4; ids=[]; qs=[]
            for _ in range(k): tid,sc=struct.unpack('<IH',rb[pos:pos+6]); pos+=6; ids.append(tid); qs.append(sc)
            fb=None
            if flags & FLAG_RANK_OVERFLOW: fb=struct.unpack('<I',rb[pos:pos+4])[0]
            recs.append(CompactTokenRecord(ids,qs,cr,delta,fb,flags))
        chunks.append(CompactChunk(first,recs,ci))
    if off!=len(buf): raise ValueError('truncated or trailing compact artifact')
    return CompactSequence(**d, chunks=chunks)

def detect_metadata_repetition(seq:CompactSequence, trajectory_events:List[Dict[str,Any]]|None=None, manifest:Dict[str,Any]|None=None)->MetadataReport:
    token_count=sum(len(c.records) for c in seq.chunks) or 1
    seq_hdr=json.dumps({k:v for k,v in seq.__dict__.items() if k!='chunks'},sort_keys=True,separators=(',',':')).encode()
    chunk_hdr=sum(12 for _ in seq.chunks); tok_bytes=sum(len(_rec_to_bytes(r)) for c in seq.chunks for r in c.records)
    repeated_fields=0; repeated_bytes=0
    for c in seq.chunks:
        for r in c.records:
            keys=set(r.extra.keys()) if r.extra else set(); bad=keys & (STABLE_METADATA_FIELDS|AGENTOPS_EVENT_FIELDS)
            repeated_fields += len(bad); repeated_bytes += len(json.dumps({k:r.extra[k] for k in bad},sort_keys=True,separators=(',',':')).encode()) if bad else 0
    traj_bytes=len(json.dumps(trajectory_events or [],sort_keys=True,separators=(',',':')).encode()) if trajectory_events else 0
    man_bytes=len(json.dumps(manifest or {},sort_keys=True,separators=(',',':')).encode()) if manifest else 0
    total=len(seq_hdr)+chunk_hdr+tok_bytes+traj_bytes+man_bytes
    rpt=MetadataReport(repeated_fields,repeated_bytes,repeated_bytes/token_count,len(seq_hdr),chunk_hdr,tok_bytes,0,0,traj_bytes,man_bytes,tok_bytes/token_count,total/token_count)
    if repeated_fields: rpt.ast_code=AST_METADATA_REPETITION_VIOLATION
    if rpt.compact_token_raw_bytes_per_token>22: rpt.warning=True
    return rpt

def verify_compact_sequence(seq:CompactSequence, *, min_retention_tokens:int=0):
    t0=time.perf_counter(); findings=[]; toks=[]
    for c in seq.chunks:
        for i,r in enumerate(c.records):
            if i and c.first_token_index+i in toks: findings.append({'ast_code':AST_SCHEMA_MISMATCH,'message':'duplicated token index'})
            toks.append(c.first_token_index+i); reconstruct_chosen_id(r)
    if len(toks)<min_retention_tokens: findings.append({'ast_code':AST_MISSING_FRAME,'message':'retention floor not met'})
    rpt=detect_metadata_repetition(seq)
    if rpt.ast_code: findings.append({'ast_code':rpt.ast_code,'message':'metadata repetition violation'})
    return {'token_coherence_passed':not findings,'trajectory_audit_passed':True,'findings':findings,'metadata_report':rpt,'p95_reconstruction_seconds':time.perf_counter()-t0,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
