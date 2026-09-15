"""Bounded V3.3 verification. Keeps one declared, size-limited chunk in memory.

Counters describe logical parser work, including replay passes; bytes_hashed
counts bytes submitted to SHA-256 directly (HMAC work is counted separately).
No span overlay is needed for structural verification.
"""
from __future__ import annotations

import binascii
import hashlib
import json
from pathlib import Path
from . import compact_evidence as ce
from .evidence_profile import get_evidence_profile


def verify_stream(path, audit_keys=None, *, tension_maps=None,
                  max_artifact_bytes=1024 * 1024 * 1024, max_chunk_tokens=1024,
                  max_metadata_bytes=1024 * 1024, counters=None, work_bounds=None):
    """Verify V3.3 and return metadata/statistics, never a token collection.

    A repeatable token source supports the shared authenticated replay routine.
    Extra passes seek over verified span/manifest bytes rather than retaining
    tokens. The same file descriptor is held through verification.
    """
    if counters is None:
        counters = {}
    counters.update(bytes_read=0, bytes_hashed=0, token_records_decoded=0,
                    candidate_entries_decoded=0, varint_bytes_decoded=0,
                    chunks_verified=0, span_events_indexed=0,
                    span_overlay_operations=0, full_artifact_materializations=0,
                    replay_passes=0, maximum_chunk_payload_bytes=0,
                    allocations=0, maximum_live_bytes=0)
    if work_bounds is None:
        work_bounds = {}
    work_bounds.update(max_artifact_bytes=max_artifact_bytes,
                       max_metadata_bytes=max_metadata_bytes,
                       max_chunk_tokens=max_chunk_tokens, max_replay_passes=4)
    def digest(data):
        counters['bytes_hashed'] += len(data)
        return hashlib.sha256(data).digest()
    with Path(path).open('rb') as fh:
        size = fh.seek(0, 2)
        if size > max_artifact_bytes:
            raise ValueError('work bound: artifact exceeds streaming verifier absolute size limit')
        fh.seek(0)
        full_hash, content_hash = hashlib.sha256(), hashlib.sha256()
        def read(n, *, primary=True, content=True):
            if n < 0 or n > max_artifact_bytes:
                raise ValueError('invalid streaming read size')
            b = fh.read(n)
            counters['bytes_read'] += len(b)
            counters['allocations'] += 1
            counters['maximum_live_bytes'] = max(counters['maximum_live_bytes'], len(b))
            if len(b) != n:
                raise ValueError('artifact is truncated')
            if primary:
                full_hash.update(b)
                counters['bytes_hashed'] += len(b)
                if content:
                    content_hash.update(b)
                    counters['bytes_hashed'] += len(b)
            return b
        raw_header = read(ce._HEADER_V33.size)
        f = ce._HEADER_V33.unpack(raw_header)
        (magic, version, profile_enum, assurance, nonce, seq, tokenizer,
         signal_schema, signal_hash, probe, probe_hash, controls, base, maximum,
         policy, rate, key_id, commitment, tension_id, tension_hash,
         quantization, capacity, work, calibration, retention, meta_len,
         chunk_count, span_count, token_count) = f
        if magic != ce.MAGIC or version != ce.VERSION_V33:
            raise ValueError('compact evidence schema mismatch')
        profile = ce._profile_from_enum(profile_enum)
        prof = get_evidence_profile(profile)
        if assurance != 0:
            raise ValueError('software-only decoder supports DSA-R assurance class only')
        if base != prof.base_k or not (1 <= base <= maximum <= prof.max_adaptive_k):
            raise ValueError('V3.3 profile/top-k declaration mismatch')
        if not 0 <= rate <= 1000000:
            raise ValueError('invalid V3.3 stochastic sampling rate')
        if not 1 <= capacity <= max_chunk_tokens:
            raise ValueError('work bound: V3.3 chunk token capacity exceeds streaming verifier limit')
        if meta_len > max_metadata_bytes:
            raise ValueError('work bound: V3.3 metadata exceeds streaming verifier limit')
        if chunk_count > token_count or (token_count and not chunk_count):
            raise ValueError('V3.3 chunk count mismatch')
        if prof.absolute_ci_safety_rail_mib and size > prof.absolute_ci_safety_rail_mib * 1024 * 1024:
            raise ValueError('artifact exceeds profile absolute safety rail')
        for name, value in [('tokenizer_id', tokenizer), ('signal_schema_id', signal_schema),
                            ('quantization_id', quantization), ('verifier_work_profile_id', work),
                            ('runtime_calibration_id', calibration), ('retention_policy_id', retention)]:
            ce._validate_v33_header_field(name, value)
        expected_fields = prof.v33_header_fields
        if quantization != expected_fields.quantization_id:
            raise ValueError('unsupported V3.3 quantization identifier for declared profile')
        if work != expected_fields.verifier_work_profile_id:
            raise ValueError('unsupported V3.3 verifier work-profile identifier for declared profile')
        meta_hash = read(32)
        meta_bytes = read(meta_len)
        if digest(meta_bytes) != meta_hash:
            raise ValueError('V3.3 metadata digest mismatch')
        metadata = json.loads(meta_bytes.decode('utf8')) if meta_bytes else {}
        if not isinstance(metadata, dict) or metadata.get('profile_id') != profile:
            raise ValueError('V3.3 metadata profile declaration mismatch')
        # Conservative logical wire residency, independent of Python object sizes.
        # Metadata/header/manifest, two chunk buffers (read transition), and the
        # logarithmic Merkle frontier plus temporary digests are covered.
        fixed_wire_bytes = (len(raw_header) + len(meta_bytes) + 32
                            + 2 * ce._MANIFEST_V33.size + 2 * ce._CHUNK_V33.size
                            + 32 * (chunk_count.bit_length() + 4))
        counters['maximum_live_bytes'] = max(counters['maximum_live_bytes'], fixed_wire_bytes)
        token_start = fh.tell()
        merkle = []
        def add_leaf(h):
            level = 0
            while level < len(merkle) and merkle[level] is not None:
                h = digest(merkle[level] + h)
                merkle[level] = None
                level += 1
            if level == len(merkle):
                merkle.append(h)
            else:
                merkle[level] = h
        def root():
            h = None
            level_h = 0
            for level, left in enumerate(merkle):
                if left is None:
                    continue
                if h is None:
                    h, level_h = left, level
                else:
                    while level_h < level:
                        h = digest(h + h)
                        level_h += 1
                    h = digest(left + h)
                    level_h += 1
            return h if h is not None else digest(b'')
        hist = {3: 0, 5: 0, 10: 0, 255: 0}
        triggers = dict.fromkeys(ce._TRIGGER_NAMES, 0)
        bitmap = hashlib.sha256()
        stats = dict(token_count=token_count, adaptive_count=0, max_effective_topk=0,
                     rank_overflow=0, chunk_count=chunk_count, span_count=span_count)
        bits = [('rank', ce.TRIGGER_RANK), ('stochastic', ce.TRIGGER_STOCHASTIC),
                ('history', ce.TRIGGER_HISTORY), ('canary', ce.TRIGGER_CANARY),
                ('escalation', ce.TRIGGER_ESCALATION)]
        def tokens(primary):
            fh.seek(token_start)
            expected_start = 0
            for index in range(chunk_count):
                ch = ce._CHUNK_V33.unpack(read(ce._CHUNK_V33.size, primary=primary))
                ci, start, count, flags, cb, maxk, rc, sc, hc, cc, length, crc, sha = ch
                if ci != index or start != expected_start:
                    raise ValueError('compact chunks are missing or reordered')
                if flags != int(index == chunk_count - 1) or cb != base or not 1 <= count <= capacity:
                    raise ValueError('V3.3 chunk declaration mismatch')
                if length > count * (8 + 5 * maximum):
                    raise ValueError('V3.3 chunk payload exceeds bounded canonical size')
                payload = read(length, primary=primary)
                counters['maximum_chunk_payload_bytes'] = max(counters['maximum_chunk_payload_bytes'], length)
                counters['maximum_live_bytes'] = max(counters['maximum_live_bytes'], fixed_wire_bytes + 2 * length)
                actual_sha = digest(payload)
                if binascii.crc32(payload) & 0xffffffff != crc or actual_sha != sha:
                    raise ValueError('compact chunk integrity check failed')
                counters['chunks_verified'] += 1
                if primary:
                    add_leaf(sha)
                records = ce._decode_v33_chunk_payload(payload, start, count, base)
                counters['token_records_decoded'] += len(records)
                counters['candidate_entries_decoded'] += sum(r.effective_topk for r in records)
                if any(r.effective_topk > maximum for r in records):
                    raise ValueError('V3.3 token evidence exceeds declared maximum adaptive K')
                if max(r.effective_topk for r in records) != maxk:
                    raise ValueError('V3.3 maximum effective top-k mismatch')
                if [sum(bool(r.trigger_flags & bit) for r in records) for _, bit in bits[:4]] != [rc, sc, hc, cc]:
                    raise ValueError('V3.3 chunk trigger count mismatch')
                for rec in records:
                    if rec.trigger_flags & ~31 or rec.record_flags & ~ce.RECORD_HAS_FALLBACK_CHOSEN_ID:
                        raise ValueError('unknown V3.3 token flags')
                    candidate_ids = rec.topk_ids
                    if len(candidate_ids) != len(set(candidate_ids)):
                        raise ValueError('duplicate V3.3 candidate token IDs are not allowed')
                    if primary:
                        k = rec.effective_topk
                        hist[next(b for b in (3, 5, 10, 255) if k <= b)] += 1
                        active = sum(bool(rec.trigger_flags & bit) for _, bit in bits)
                        for name, bit in bits:
                            triggers[name] += bool(rec.trigger_flags & bit)
                        triggers['multi'] += active > 1
                        bitmap.update(bytes([bool(rec.trigger_flags & ce.TRIGGER_STOCHASTIC)]))
                        counters['bytes_hashed'] += 1
                        stats['adaptive_count'] += k > base
                        stats['max_effective_topk'] = max(stats['max_effective_topk'], k)
                        stats['rank_overflow'] += rec.chosen_rank == 255 or rec.chosen_rank > maximum
                    yield rec
                del records, payload
                expected_start += count
            if expected_start != token_count:
                raise ValueError('missing token evidence')
        for _ in tokens(True):
            pass
        for _ in range(span_count):
            start, end, _, _, _ = ce._SPAN_V33.unpack(read(ce._SPAN_V33.size))
            flags = read(1)[0]
            if flags & ~ce.SPAN_HAS_EVALUATOR_ID:
                raise ValueError('unknown V3.3 span flags')
            if flags & ce.SPAN_HAS_EVALUATOR_ID:
                read(ce._SPAN_V33_EVAL.size)
            if start >= end or end > token_count:
                raise ValueError('sparse span is outside token range')
            counters['span_events_indexed'] += 1
        raw_manifest = read(ce._MANIFEST_V33.size, content=False)
        if fh.tell() != size:
            raise ValueError('unexpected trailing or missing V3.3 manifest bytes')
        content_hash.update(ce._manifest_with_zero_hash(raw_manifest))
        counters['bytes_hashed'] += len(raw_manifest)
        fields = ce._MANIFEST_V33.unpack(raw_manifest)
        manifest = ce._manifest_from_fields(fields)
        if (manifest.token_count, manifest.chunk_count, manifest.span_event_count) != (token_count, chunk_count, span_count):
            raise ValueError('V3.3 manifest count mismatch')
        if manifest.sequence_header_hash != digest(raw_header).hex():
            raise ValueError('V3.3 sequence header hash mismatch')
        if manifest.chunk_merkle_root != root().hex():
            raise ValueError('V3.3 chunk Merkle root mismatch')
        if manifest.artifact_content_hash != content_hash.hexdigest():
            raise ValueError('V3.3 artifact content hash mismatch')
        if manifest.raw_evidence_bytes != size or manifest.minimum_reconstructable_bytes != size:
            raise ValueError('V3.3 reconstructable byte floor/count mismatch')
        if manifest.effective_k_histogram != hist or manifest.trigger_count_by_reason != triggers:
            raise ValueError('V3.3 manifest histogram/trigger mismatch')
        if manifest.audit_selection_digest != bitmap.hexdigest() or fields[19] != 0:
            raise ValueError('V3.3 manifest audit outcome/digest mismatch')
        header = ce.MonologueSequenceHeaderV3(seq, token_count, profile, base, maximum, capacity, schema_version=version)
        meta = {**metadata, 'audit_key_id': key_id, 'audit_selection_commitment': commitment.hex(),
                'policy_version': policy, 'stochastic_rate_ppm': rate,
                'prompt_nonce': metadata.get('prompt_nonce', nonce), 'header_prompt_nonce': nonce,
                'tension_map_id': tension_id, 'tension_map_hash': tension_hash.hex()}
        class ReplayTokens:
            def __iter__(self):
                counters['replay_passes'] += 1
                if counters['replay_passes'] > work_bounds['max_replay_passes']:
                    raise ValueError('work bound: replay pass limit exceeded')
                return tokens(False)
        decoded = dict(header=header, meta=meta, manifest=manifest, tokens=ReplayTokens())
        keyed = rate != 0 or commitment != ce.ZERO_HASH or bool(audit_keys)
        if keyed:
            def count_hash(n):
                counters['bytes_hashed'] += n
            ce.verify_keyed_replay(decoded, audit_keys or {}, tension_maps=tension_maps,
                                   hash_observer=count_hash)
        decoded.pop('tokens')
        decoded.update(sha256=full_hash.hexdigest(), raw_bytes=size,
                       minimum_reconstructable_bytes=size, stats=stats, counters=counters,
                       work_bounds=work_bounds)
        return decoded