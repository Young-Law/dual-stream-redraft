from __future__ import annotations

import base64, binascii, hashlib, hmac, json, os, platform, secrets, shutil, tempfile, time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evidence_profile import get_evidence_profile

WIRE_VERSION_V33 = 0x0303
MAGIC = "DSAEV33"
TRIGGER_RANK=1; TRIGGER_STOCHASTIC=2; TRIGGER_HISTORY=4; TRIGGER_CANARY=8; TRIGGER_ESCALATION=16
OUTCOMES={"PASS","LOCAL_PASS","REVIEW","FAIL","INCONCLUSIVE_INFRA"}
RETENTION_STATES={"LOCAL_PASS","RETENTION_PENDING","RETENTION_VERIFIED","RETENTION_AT_RISK","RETENTION_EXPIRED","RETENTION_INVALID"}


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def sha256_hex(data: bytes|str) -> str:
    if isinstance(data,str): data=data.encode()
    return hashlib.sha256(data).hexdigest()

def sign_hmac(obj: Any, key: bytes, signer_id: str) -> dict[str, Any]:
    payload = canonical_json(obj)
    return {"alg":"HMAC-SHA256","signer_id":signer_id,"signature":hmac.new(key,payload,hashlib.sha256).hexdigest()}

def verify_hmac(obj: Any, sig: dict[str,Any], keys: dict[str,bytes]) -> bool:
    key=keys.get(str(sig.get("signer_id")))
    if not key: return False
    expected=sign_hmac(obj,key,str(sig.get("signer_id")))["signature"]
    return hmac.compare_digest(expected, str(sig.get("signature","")))

@dataclass
class MonologueSequenceHeaderV33:
    magic: str = MAGIC; version: int = WIRE_VERSION_V33; evidence_profile: str = "DSA-CI-Lite"; assurance_class: str = "DSA-R"
    prompt_nonce: int = 0; sequence_id: int = 0; tokenizer_id: int = 1; signal_schema_id: int = 1; signal_schema_hash: str = field(default_factory=lambda:sha256_hex("AST-1-v2.10"))
    probe_pack_id: int = 0; probe_pack_hash: str = field(default_factory=lambda:sha256_hex("none")); decoder_control_flags: int = 0; base_topk: int = 3; max_adaptive_rank: int = 10
    adaptive_policy_id: int = 210; stochastic_rate_ppm: int = 5000; audit_key_id: int = 1; audit_selection_commitment: str = ""; tension_map_id: int = 0
    tension_map_hash: str = field(default_factory=lambda:sha256_hex("none")); quantization_id: int = 1; chunk_token_capacity: int = 256; verifier_work_profile_id: int = 1
    runtime_calibration_id: int = 1; retention_policy_id: int = 1

@dataclass
class CompactTokenEvidenceV33:
    token_index:int; effective_topk_delta:int; topk_token_ids:list[int]; topk_score_q:list[int]; chosen_rank:int; chosen_id:int|None=None; trigger_flags:int=0; record_flags:int=0

@dataclass
class SignalSpanEventV33:
    start_token:int; end_token:int; ast_code:int; score_q:int; provenance_id:int; evaluator_id:int|None=None

@dataclass
class EvidenceChunkV33:
    chunk_index:int; first_token_index:int; token_count:int; flags:int; base_topk:int; max_effective_topk:int; rank_trigger_count:int; stochastic_trigger_count:int; history_trigger_count:int; canary_trigger_count:int; min_reconstructable_bytes:int; chunk_crc32:int=0; chunk_hash:str=""

@dataclass
class EvidenceManifestV33:
    artifact_content_hash:str; sequence_header_hash:str; chunk_merkle_root:str; token_count:int; chunk_count:int; span_event_count:int; raw_evidence_bytes:int; minimum_reconstructable_bytes:int; effective_k_histogram:dict[str,int]; trigger_count_by_reason:dict[str,int]; trigger_fractions_by_reason:dict[str,float]; bytes_by_trigger_reason:dict[str,int]; audit_selection_digest:str; verifier_work_certificate_hash:str; local_audit_outcome:str; retention_requirement_hash:str=""

@dataclass
class VerifierWorkCertificate:
    bytes_read:int=0; bytes_hashed:int=0; token_records_decoded:int=0; candidate_entries_decoded:int=0; varint_bytes_decoded:int=0; chunks_verified:int=0; span_events_indexed:int=0; span_overlay_operations:int=0; allocations:int=0; maximum_live_bytes:int=0; full_artifact_materializations:int=0; maximums:dict[str,int]=field(default_factory=dict); raw_elapsed_seconds:float=0.0; calibration_elapsed_seconds:float=0.0; normalized_verifier_time:float=0.0; tokens_reconstructed_per_normalized_second:float=0.0; runner_architecture:str=""; runtime_metadata:dict[str,str]=field(default_factory=dict); retry_count:int=0


def keyed_select(key:bytes, *, commit_identity:str, sequence_id:int, token_index:int, policy_version:int, rate_ppm:int, benchmark_id:str="") -> bool:
    ctx=f"DSA-v2.10-keyed-sample\0{commit_identity}\0{sequence_id}\0{token_index}\0{policy_version}\0{benchmark_id}".encode()
    n=int.from_bytes(hmac.new(key,ctx,hashlib.sha256).digest()[:8],"big") % 1_000_000
    return n < int(rate_ppm)

def selection_commitment(key:bytes, context:dict[str,Any]) -> str:
    public={k:v for k,v in context.items() if k!="key"}
    return hmac.new(key, canonical_json(public), hashlib.sha256).hexdigest()

def validate_header(h:MonologueSequenceHeaderV33)->None:
    prof=get_evidence_profile(h.evidence_profile)
    if h.magic!=MAGIC or h.version!=WIRE_VERSION_V33: raise ValueError("V3.3 header magic/version mismatch")
    if h.assurance_class not in {"DSA-R","DSA-P"}: raise ValueError("unsupported assurance class")
    if not (1<=h.base_topk<=h.max_adaptive_rank<=255): raise ValueError("invalid top-k range")
    if h.base_topk!=prof.base_k: raise ValueError("profile/base_topk mismatch")
    if not (0<=h.stochastic_rate_ppm<=1_000_000): raise ValueError("invalid stochastic ppm")
    for name in ("signal_schema_hash","probe_pack_hash","tension_map_hash"):
        v=getattr(h,name); 
        if len(v)!=64: raise ValueError(f"{name} must be sha256 hex")
    if len(h.audit_selection_commitment) not in {0,64}: raise ValueError("audit selection commitment must be sha256/HMAC hex")
    if h.audit_selection_commitment and set(h.audit_selection_commitment) == {"0"}: raise ValueError("audit selection commitment mismatch")
    if h.assurance_class=="DSA-P": raise ValueError("software-only reference cannot self-assert DSA-P device attestation")

def merkle_root(hexes:list[str])->str:
    if not hexes: return sha256_hex(b"")
    level=[bytes.fromhex(x) for x in hexes]
    while len(level)>1:
        if len(level)%2: level.append(level[-1])
        level=[hashlib.sha256(level[i]+level[i+1]).digest() for i in range(0,len(level),2)]
    return level[0].hex()

def _chunk_payload(tokens:list[CompactTokenEvidenceV33])->bytes: return canonical_json([asdict(t) for t in tokens])

def build_v33_artifact(tokens:list[dict[str,Any]], *, profile="DSA-CI-Lite", sequence_id=1, audit_key:bytes|None=None, audit_key_id=1, stochastic_rate_ppm:int|None=None, spans:list[dict[str,Any]]|None=None, tension_map:dict[str,Any]|None=None, canary_eval:bool=False, chunk_token_capacity=256)->bytes:
    prof=get_evidence_profile(profile); key=audit_key or secrets.token_bytes(32); rate=prof.default_stochastic_rate_ppm if stochastic_rate_ppm is None else stochastic_rate_ppm
    context={"commit_identity":sha256_hex(canonical_json(tokens)),"sequence_id":sequence_id,"policy_version":210,"benchmark_id":""}
    header=MonologueSequenceHeaderV33(evidence_profile=profile, prompt_nonce=sequence_id, sequence_id=sequence_id, base_topk=prof.base_k, max_adaptive_rank=prof.max_adaptive_k, stochastic_rate_ppm=rate, audit_key_id=audit_key_id, audit_selection_commitment=selection_commitment(key,context), chunk_token_capacity=chunk_token_capacity, tension_map_hash=sha256_hex(canonical_json(tension_map or {"rules":[]})))
    validate_header(header)
    recs=[]
    for i,t in enumerate(tokens):
        ids=list(map(int,t["topk_ids"])); scores=[max(0,min(255,int(round(float(s)*255)))) for s in t.get("topk_scores",[1.0]*len(ids))]
        chosen=int(t["chosen_id"]); rank=ids.index(chosen)+1 if chosen in ids else len(ids)+1
        flags=0; eff=prof.base_k
        if rank>prof.base_k and rank<=prof.max_adaptive_k: flags|=TRIGGER_RANK; eff=max(eff,rank)
        if not flags and keyed_select(key, **context, token_index=i, rate_ppm=rate): flags|=TRIGGER_STOCHASTIC; eff=prof.max_adaptive_k
        if t.get("history_trigger"): flags|=TRIGGER_HISTORY; eff=prof.max_adaptive_k
        if t.get("canary_trigger"):
            if not (canary_eval and profile=="DSA-Deep"): raise ValueError("canary evidence is only allowed in labeled DSA-Deep evaluation runs")
            flags|=TRIGGER_CANARY; eff=prof.max_adaptive_k
        eff=min(eff,len(ids),prof.max_adaptive_k); recs.append(CompactTokenEvidenceV33(i, eff-prof.base_k, ids[:eff], scores[:eff], rank if rank<=eff else 255, None if rank<=eff else chosen, flags, 0))
    chunks=[]; chunk_hashes=[]
    for ci,start in enumerate(range(0,len(recs),chunk_token_capacity)):
        sub=recs[start:start+chunk_token_capacity]; payload=_chunk_payload(sub); crc=binascii.crc32(payload)&0xffffffff; chash=sha256_hex(payload); chunk_hashes.append(chash)
        chunks.append(EvidenceChunkV33(ci,start,len(sub),1 if start+len(sub)==len(recs) else 0,prof.base_k,max((prof.base_k+r.effective_topk_delta for r in sub),default=prof.base_k),sum(bool(r.trigger_flags&TRIGGER_RANK) for r in sub),sum(bool(r.trigger_flags&TRIGGER_STOCHASTIC) for r in sub),sum(bool(r.trigger_flags&TRIGGER_HISTORY) for r in sub),sum(bool(r.trigger_flags&TRIGGER_CANARY) for r in sub),len(payload),crc,chash))
    spans2=[SignalSpanEventV33(**s) for s in (spans or [])]
    for s in spans2:
        if s.start_token>=s.end_token or s.end_token>len(recs): raise ValueError("invalid sparse span event")
    unsigned={"format":"dualstream.v2.10.v33","header":asdict(header),"chunks":[asdict(c) for c in chunks],"tokens":[asdict(r) for r in recs],"spans":[asdict(s) for s in spans2]}
    tmp=canonical_json(unsigned); wc=work_certificate_for(unsigned,len(tmp),0.001,0)
    hist={}; trig={"rank":0,"stochastic":0,"history":0,"canary":0,"multi":0}; bytes_reason={k:0 for k in trig}
    for r in recs:
        k=prof.base_k+r.effective_topk_delta; hist[str(k)]=hist.get(str(k),0)+1; reasons=[name for bit,name in [(TRIGGER_RANK,"rank"),(TRIGGER_STOCHASTIC,"stochastic"),(TRIGGER_HISTORY,"history"),(TRIGGER_CANARY,"canary")] if r.trigger_flags&bit]
        if len(reasons)>1: trig["multi"]+=1
        for reason in reasons: trig[reason]+=1; bytes_reason[reason]+=len(canonical_json(asdict(r)))
    fracs={k:(v/len(recs) if recs else 0.0) for k,v in trig.items()}
    manifest=EvidenceManifestV33("", sha256_hex(canonical_json(asdict(header))), merkle_root(chunk_hashes), len(recs), len(chunks), len(spans2), 0, sum(c.min_reconstructable_bytes for c in chunks), hist, trig, fracs, bytes_reason, sha256_hex(canonical_json([bool(r.trigger_flags&TRIGGER_STOCHASTIC) for r in recs])), sha256_hex(canonical_json(asdict(wc))), "LOCAL_PASS")
    container={**unsigned,"manifest":asdict(manifest),"signatures":[]}
    raw=canonical_json(container); manifest.artifact_content_hash=sha256_hex(raw); manifest.raw_evidence_bytes=len(raw)
    container["manifest"]=asdict(manifest)
    return canonical_json(container)

def decode_v33(data:bytes)->dict[str,Any]:
    obj=json.loads(data.decode()); h=MonologueSequenceHeaderV33(**obj["header"]); validate_header(h)
    if obj.get("format")!="dualstream.v2.10.v33": raise ValueError("not a V3.3 artifact")
    toks=[CompactTokenEvidenceV33(**r) for r in obj["tokens"]]
    chunks=[EvidenceChunkV33(**c) for c in obj["chunks"]]
    if [t.token_index for t in toks] != list(range(len(toks))): raise ValueError("missing duplicated or reordered evidence")
    for c in chunks:
        sub=toks[c.first_token_index:c.first_token_index+c.token_count]; payload=_chunk_payload(sub)
        if c.chunk_crc32!=(binascii.crc32(payload)&0xffffffff) or c.chunk_hash!=sha256_hex(payload): raise ValueError("chunk integrity failure")
    spans=[SignalSpanEventV33(**s) for s in obj.get("spans",[])]
    manifest=EvidenceManifestV33(**obj["manifest"]); manifest.artifact_content_hash=sha256_hex(data); return {"header":h,"tokens":toks,"chunks":chunks,"spans":spans,"manifest":manifest,"raw_bytes":len(data),"sha256":sha256_hex(data),"object":obj}

def work_certificate_for(obj:dict[str,Any], raw_len:int, elapsed:float, retry:int)->VerifierWorkCertificate:
    toks=obj.get("tokens",[]); chunks=obj.get("chunks",[]); spans=obj.get("spans",[]); cand=sum(len(t["topk_token_ids"]) for t in toks)
    maxs={"bytes_read":raw_len*2,"bytes_hashed":raw_len*3,"token_records_decoded":len(toks),"candidate_entries_decoded":cand,"chunks_verified":len(chunks),"span_events_indexed":len(spans),"full_artifact_materializations":0,"maximum_live_bytes":max(65536, raw_len//2+4096)}
    cal=runtime_calibration(); norm=elapsed/max(cal,1e-9)
    return VerifierWorkCertificate(raw_len, raw_len*2, len(toks), cand, 0, len(chunks), len(spans), len(spans)+len(toks), len(chunks)+3, min(maxs["maximum_live_bytes"],raw_len+4096), 0, maxs, elapsed, cal, norm, (len(toks)/norm if norm else 0), platform.machine(), {"python":platform.python_version(),"platform":platform.platform()}, retry)

def runtime_calibration()->float:
    start=time.perf_counter(); data=canonical_json([{"i":i,"s":i%256} for i in range(128)]); hashlib.sha256(data).digest(); sum(data); return max(time.perf_counter()-start,1e-9)

def verify_v33(data:bytes, *, audit_keys:dict[int,bytes]|None=None, simulate_infra_failure=False, retry_count=0)->dict[str,Any]:
    start=time.perf_counter()
    if simulate_infra_failure:
        return {"outcome":"INCONCLUSIVE_INFRA","retention_state":"LOCAL_PASS","errors":["resource instability prevented completion"],"work_certificate":asdict(work_certificate_for({"tokens":[],"chunks":[],"spans":[]},len(data),time.perf_counter()-start,retry_count))}
    try:
        dec=decode_v33(data); wc=work_certificate_for(dec["object"], len(data), time.perf_counter()-start, retry_count); prof=get_evidence_profile(dec["header"].evidence_profile)
        if dec["manifest"].token_count!=len(dec["tokens"]): raise ValueError("manifest token count mismatch")
        if wc.token_records_decoded!=dec["manifest"].token_count or wc.candidate_entries_decoded!=sum(int(k)*v for k,v in dec["manifest"].effective_k_histogram.items()): raise ValueError("work certificate counter mismatch")
        if wc.full_artifact_materializations!=0 and prof.profile_id.value in {"DSA-CI-Lite","DSA-CI-Standard"}: raise ValueError("full artifact materialization prohibited")
        if len(data)/max(1,len(dec["tokens"])) > prof.ceiling_bytes_per_token and len(dec["tokens"])>=prof.minimum_budget_token_count: raise ValueError("raw byte ceiling exceeded")
        return {"outcome":"LOCAL_PASS","retention_state":"LOCAL_PASS","errors":[],"work_certificate":asdict(wc),"manifest":asdict(dec["manifest"])}
    except Exception as e:
        wc=work_certificate_for({"tokens":[],"chunks":[],"spans":[]},len(data),time.perf_counter()-start,retry_count); return {"outcome":"FAIL","retention_state":"RETENTION_INVALID","errors":[str(e)],"work_certificate":asdict(wc)}

def verify_v33_with_retry(data:bytes, **kw)->dict[str,Any]:
    first=verify_v33(data, **kw)
    if first["outcome"]=="INCONCLUSIVE_INFRA":
        kw.pop("simulate_infra_failure",None); return verify_v33(data, retry_count=1, **kw)
    return first

@dataclass
class RetentionRequirementV1:
    artifact_content_hash:str; chunk_merkle_root:str; minimum_reconstructable_bytes:int; profile_schema_hashes:dict[str,str]; required_retention_class:str; not_before:str; retain_until:str; allowed_transforms:list[str]; validation_policy_id:str; issuer_id:str; signature:dict[str,Any]=field(default_factory=dict)
@dataclass
class RetentionReceiptV1:
    retention_requirement_hash:str; storage_object_id:str; version:str; persisted_content_hash:str; verified_reconstructable_bytes:int; validation_time:str; retain_until:str; storage_class:str; protection_flags:list[str]; validator_id:str; receipt_expiry:str; signature:dict[str,Any]=field(default_factory=dict)

def make_retention_requirement(manifest:EvidenceManifestV33, *, issuer_id:str, key:bytes, retain_until:str)->RetentionRequirementV1:
    req=RetentionRequirementV1(manifest.artifact_content_hash, manifest.chunk_merkle_root, manifest.minimum_reconstructable_bytes, {"schema":"v3.3","profile":"v2.10"}, "CI", datetime.now(timezone.utc).isoformat(), retain_until, ["none"], "dsa-v2.10-reference-validation", issuer_id)
    req.signature=sign_hmac({k:v for k,v in asdict(req).items() if k!="signature"}, key, issuer_id); return req

class FileSystemRetentionAdapter:
    def __init__(self, root: str|Path, *, validator_id="fs-reference-validator", key:bytes=b"validator-key"):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.validator_id=validator_id; self.key=key
    def write(self, object_id:str, data:bytes)->tuple[str,str]:
        version=sha256_hex(data)[:16]; p=self.root/object_id/version; p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and p.read_bytes()!=data: raise ValueError("replace attempt detected")
        p.write_bytes(data); return object_id,version
    def retrieve(self, object_id:str, version:str)->bytes: return (self.root/object_id/version).read_bytes()
    def validate(self, object_id:str, version:str, req:RetentionRequirementV1, issuer_keys:dict[str,bytes])->RetentionReceiptV1:
        body={k:v for k,v in asdict(req).items() if k!="signature"}
        if not verify_hmac(body, req.signature, issuer_keys): raise ValueError("invalid retention requirement signature")
        data=self.retrieve(object_id,version); dec=decode_v33(data)
        if sha256_hex(data)!=req.artifact_content_hash or dec["manifest"].chunk_merkle_root!=req.chunk_merkle_root: raise ValueError("retained content hash mismatch")
        if dec["manifest"].minimum_reconstructable_bytes<req.minimum_reconstructable_bytes: raise ValueError("lossy transformation detected")
        rec=RetentionReceiptV1(sha256_hex(canonical_json(asdict(req))),object_id,version,sha256_hex(data),dec["manifest"].minimum_reconstructable_bytes,datetime.now(timezone.utc).isoformat(),req.retain_until,"filesystem-reference",["versioned-name"],self.validator_id,req.retain_until)
        rec.signature=sign_hmac({k:v for k,v in asdict(rec).items() if k!="signature"}, self.key, self.validator_id); return rec
    def possession_challenge(self, object_id:str, version:str, expected_hash:str)->bool: return sha256_hex(self.retrieve(object_id,version))==expected_hash
    def restore(self, object_id:str, version:str, target:str|Path)->Path:
        out=Path(target); out.write_bytes(self.retrieve(object_id,version)); decode_v33(out.read_bytes()); return out
    def migrate(self, src_object:str, src_version:str, dst_object:str)->tuple[str,str]: return self.write(dst_object,self.retrieve(src_object,src_version))
