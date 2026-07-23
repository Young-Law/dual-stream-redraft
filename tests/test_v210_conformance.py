import json, subprocess, sys
from datetime import datetime, timedelta, timezone

import pytest

from dualstream.compact_evidence import VERSION_V31, VERSION_V32, VERSION_V33, decode_compact_sequence, encode_compact_sequence
from dualstream.v210 import *

KEY=b"k"*32

def tokens(n=32):
    out=[]
    for i in range(n):
        ids=list(range(1000+i*20,1010+i*20)); chosen=ids[7] if i%9==0 else ids[0]
        out.append({"chosen_id":chosen,"topk_ids":ids,"topk_scores":[1-j/20 for j in range(10)],"history_trigger":i==5})
    return out

def test_v33_round_trip_and_manifest_work_certificate():
    art=build_v33_artifact(tokens(), audit_key=KEY, sequence_id=7)
    dec=decode_v33(art)
    assert dec["header"].version==0x0303
    assert dec["manifest"].token_count==32
    assert dec["manifest"].trigger_count_by_reason["rank"] >= 1
    report=verify_v33(art)
    assert report["outcome"]=="LOCAL_PASS"
    wc=report["work_certificate"]
    assert wc["token_records_decoded"]==32
    assert wc["candidate_entries_decoded"]==sum(int(k)*v for k,v in report["manifest"]["effective_k_histogram"].items())
    assert wc["chunks_verified"]==report["manifest"]["chunk_count"]
    assert wc["full_artifact_materializations"]==0

def test_legacy_dispatch_isolated():
    legacy=encode_compact_sequence([{"chosen_id":1,"topk_ids":[1,2,3],"topk_scores":[1,.5,.2]}], adaptive_k=False)
    assert decode_compact_sequence(legacy)["header"].schema_version==VERSION_V32
    art=build_v33_artifact(tokens(1), audit_key=KEY)
    assert decode_compact_sequence(art)["header"].schema_version==VERSION_V33
    bad=bytearray(legacy); bad[8:10]=(0x9999).to_bytes(2,"little")
    with pytest.raises(ValueError): decode_compact_sequence(bytes(bad))

def test_keyed_sampling_exact_replay_and_commitment_rejection():
    ctx={"commit_identity":"abc","sequence_id":1,"policy_version":210,"benchmark_id":"b"}
    bitmap=[keyed_select(KEY, **ctx, token_index=i, rate_ppm=500000) for i in range(64)]
    assert bitmap == [keyed_select(KEY, **ctx, token_index=i, rate_ppm=500000) for i in range(64)]
    assert bitmap != [keyed_select(b"x"*32, **ctx, token_index=i, rate_ppm=500000) for i in range(64)]
    assert selection_commitment(KEY,ctx) != selection_commitment(b"x"*32,ctx)
    art=build_v33_artifact(tokens(4), audit_key=KEY)
    assert b"kkkk" not in art
    obj=json.loads(art); obj["header"]["audit_selection_commitment"]="0"*64
    with pytest.raises(ValueError): decode_v33(canonical_json(obj))

def test_canary_only_deep_labeled():
    t=tokens(1); t[0]["canary_trigger"]=True
    with pytest.raises(ValueError): build_v33_artifact(t, audit_key=KEY, profile="DSA-CI-Lite")
    assert decode_v33(build_v33_artifact(t, audit_key=KEY, profile="DSA-Deep", canary_eval=True))["tokens"][0].trigger_flags & TRIGGER_CANARY

def test_infra_retry_never_passes_incomplete():
    art=build_v33_artifact(tokens(2), audit_key=KEY)
    r=verify_v33(art, simulate_infra_failure=True)
    assert r["outcome"]=="INCONCLUSIVE_INFRA"
    assert verify_v33_with_retry(art, simulate_infra_failure=True)["outcome"]=="LOCAL_PASS"

def test_retention_requirement_receipt_storage_failures(tmp_path):
    art=build_v33_artifact(tokens(8), audit_key=KEY)
    dec=decode_v33(art); issuer_key=b"issuer"
    req=make_retention_requirement(dec["manifest"], issuer_id="issuer", key=issuer_key, retain_until=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat())
    store=FileSystemRetentionAdapter(tmp_path)
    oid,ver=store.write("a", art)
    receipt=store.validate(oid,ver,req,{"issuer":issuer_key})
    assert verify_hmac({k:v for k,v in asdict(receipt).items() if k!="signature"}, receipt.signature, {store.validator_id:store.key})
    assert store.possession_challenge(oid,ver,sha256_hex(art))
    restored=store.restore(oid,ver,tmp_path/"restore.json")
    assert restored.read_bytes()==art
    (tmp_path/oid/ver).write_bytes(art[:-10])
    with pytest.raises(Exception): store.validate(oid,ver,req,{"issuer":issuer_key})

def test_cli_conformance():
    cp=subprocess.run([sys.executable,"-m","dualstream.cli","conformance","v2.10","--json"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert cp.returncode==0, cp.stderr+cp.stdout
    assert json.loads(cp.stdout)["wire_version"]=="0x0303"
