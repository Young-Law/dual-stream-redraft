# DSA v2.10 Phase 1 scope

The repository implements the compact binary V3.3 evidence wire foundation, integrated model-generation path, legacy decoding compatibility, canonical artifact hashing, and authorized keyed-selection replay. Portable verifier governance, signed tension-map governance, and complete end-to-end retention assurance remain under implementation.

V3.1 and V3.2 artifacts remain legacy artifacts and are decoded by their version-specific binary decoders. They are not silently upgraded or reinterpreted as V3.3. DSA-P claims require external device-bound evidence and independent authorized retention receipts.

## Local V3.3 streaming verification

`verify_evidence_artifact` dispatches V3.3 to a streaming parser. It retains a
bounded chunk and a logarithmic Merkle frontier, validates sparse spans without
constructing token/span overlays, and replays keyed selection using at most four
additional token passes. The local limits are 1 GiB per artifact (also subject to
the profile's absolute safety rail), 1,024 tokens per chunk, and a configurable
metadata cap. The wire format already limits embedded metadata to 65,535 bytes;
the adjacent `meta.json` file is capped at 1 MiB. Chunk payload lengths are
checked against the declared token count and maximum candidate width before read.

Work certificates report parser reads, direct SHA-256 input bytes, completed
record/candidate decodes, checked chunks and spans, and full materializations.
Replay work is included. SHA-256 input accounting excludes HMAC internals,
calibration, and adjacent run metadata. Allocation counts are unavailable (`null`);
traced peak bytes remain a Python runtime observation. Failed checks preserve
completed operation counters with `work_completed=false`; these are lower bounds
on interrupted work, not a certificate of successful verification. Signatures
cover this completion flag and the declared work bounds.

Completed checks exceeding timing or observed-memory SLOs produce warnings.
Actual memory/timeout interruption produces `INCONCLUSIVE_INFRA`, never
`LOCAL_PASS`; deterministic parser limits fail with code 523. The default retry
policy permits one retry. Calibration reports a dimensionless ratio against a
fixed local candidate-decoding/hash fixture, not normalized runtime seconds.
Legacy V3.1/V3.2 checks retain materialized compatibility behavior and explicitly
report that distinction. Semantic audit and independent retention attestation
remain outside the local verifier's success claim.

The verifier's compatibility report keeps runtime observations for callers, but
its signed payload excludes traced peak and runtime fields and binds the verified
artifact's final SHA-256. Its `v33_streaming_work` report is distinct from the
standalone version-1 certificate in `dualstream.work_certificate`: registered
work-envelope integration and cross-host conformance remain pending. Neither
report authenticates a semantic audit result.
