"""DSA v2.10 Streaming-Space Envelopes (§5.3).

A streaming-space envelope wraps incremental evidence as it is produced
token-by-token, carrying session metadata, interim commitment hashes,
and retention policy bindings. This enables evidence integrity verification
in online/streaming contexts where the full artifact is not available
until generation completes.

The envelope lifecycle:

    1. ``EnvelopeSession.open()``  — create a new streaming session
    2. ``session.ingest(token_record)``  — add one token's evidence
    3. ``session.interim_commitment()``  — get a checkpoint hash
    4. ``session.seal()``  — finalize, producing a ``SealedEnvelope``
    5. ``verify_envelope(sealed, artifact_bytes)``  — verify post-hoc

Canonical JSON serialization and HMAC-SHA256 signatures bind every
state transition to the session authority's key.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EnvelopeState(str, Enum):
    OPEN = "open"
    SEALED = "sealed"
    EXPIRED = "expired"


@dataclass(frozen=True)
class EnvelopePolicy:
    """Policy governing envelope behavior."""
    max_token_count: int = 100_000
    max_interim_commits: int = 100
    commit_interval: int = 100  # auto-commit every N tokens
    ttl_seconds: float = 3600.0
    require_signature: bool = True
    enforce_canonical_order: bool = True


@dataclass
class InterimCommitment:
    """A checkpoint commitment at a given token index."""
    token_index: int
    cumulative_hash: str  # SHA-256 hex of all evidence up to this point
    token_count: int
    timestamp: float
    nonce: str = ""

    def to_canonical_json(self) -> str:
        return json.dumps(
            {"token_index": self.token_index, "cumulative_hash": self.cumulative_hash,
             "token_count": self.token_count, "timestamp": self.timestamp, "nonce": self.nonce},
            sort_keys=True, separators=(",", ":"),
        )


@dataclass
class EnvelopeHeader:
    """Metadata carried by the envelope from session start."""
    session_id: str
    profile_id: str
    issuer_id: str
    retention_policy_id: str
    opened_at: float
    assurance_class: str
    benchmark_id: str = ""
    prompt_hash: str = ""
    base_k: int = 3
    max_adaptive_k: int = 10
    adaptive_policy: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_canonical_json(self) -> str:
        payload = {
            "session_id": self.session_id,
            "profile_id": self.profile_id,
            "issuer_id": self.issuer_id,
            "retention_policy_id": self.retention_policy_id,
            "opened_at": self.opened_at,
            "assurance_class": self.assurance_class,
            "benchmark_id": self.benchmark_id,
            "prompt_hash": self.prompt_hash,
            "base_k": self.base_k,
            "max_adaptive_k": self.max_adaptive_k,
            "adaptive_policy": self.adaptive_policy,
        }
        # Include meta keys in sorted order for determinism
        if self.meta:
            payload["meta"] = self.meta
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class SealedEnvelope:
    """The final, immutable sealed envelope produced when a session closes.

    Contains the full header, all interim commitments, the final cumulative
    hash, the sealed artifact hash, and optional signatures.
    """
    header: EnvelopeHeader
    final_cumulative_hash: str
    sealed_at: float
    total_tokens: int
    interim_commitments: List[InterimCommitment]
    artifact_content_hash: str = ""  # SHA-256 of the final sealed artifact
    signature: bytes = b""
    seal_nonce: str = ""

    def to_canonical_json(self) -> str:
        payload = {
            "header_json": self.header.to_canonical_json(),
            "final_cumulative_hash": self.final_cumulative_hash,
            "sealed_at": self.sealed_at,
            "total_tokens": self.total_tokens,
            "interim_commitments": [ic.to_canonical_json() for ic in self.interim_commitments],
            "artifact_content_hash": self.artifact_content_hash,
            "seal_nonce": self.seal_nonce,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

    def hash(self) -> bytes:
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).digest()


class EnvelopeSession:
    """A mutable session for building a streaming-space envelope.

    Usage::

        session = EnvelopeSession.open(
            session_id="sess-001",
            profile_id="DSA-CI-Lite",
            issuer_id="verifier-1",
            retention_policy_id="local-floor-v1",
            issuer_key=b"secret",
        )
        for record in token_records:
            session.ingest(record)
        sealed = session.seal(final_artifact_bytes)
    """

    def __init__(
        self,
        header: EnvelopeHeader,
        policy: EnvelopePolicy,
        issuer_key: Optional[bytes],
    ) -> None:
        self._header = header
        self._policy = policy
        self._issuer_key = issuer_key
        self._state = EnvelopeState.OPEN
        self._hasher = hashlib.sha256()
        self._token_count = 0
        self._last_token_index = -1
        self._interim_commitments: List[InterimCommitment] = []
        self._auto_commit_counter = 0
        # Seed the hasher with the header
        self._hasher.update(header.to_canonical_json().encode("utf-8"))

    @classmethod
    def open(
        cls,
        *,
        session_id: str,
        profile_id: str = "DSA-CI-Lite",
        issuer_id: str = "",
        retention_policy_id: str = "local-floor-v1",
        issuer_key: Optional[bytes] = None,
        assurance_class: str = "DSA-R",
        benchmark_id: str = "",
        prompt_hash: str = "",
        base_k: int = 3,
        max_adaptive_k: int = 10,
        adaptive_policy: str = "",
        meta: Optional[Dict[str, Any]] = None,
        policy: Optional[EnvelopePolicy] = None,
    ) -> "EnvelopeSession":
        """Create and open a new envelope session."""
        header = EnvelopeHeader(
            session_id=session_id,
            profile_id=profile_id,
            issuer_id=issuer_id,
            retention_policy_id=retention_policy_id,
            opened_at=time.time(),
            assurance_class=assurance_class,
            benchmark_id=benchmark_id,
            prompt_hash=prompt_hash,
            base_k=base_k,
            max_adaptive_k=max_adaptive_k,
            adaptive_policy=adaptive_policy,
            meta=meta or {},
        )
        pol = policy or EnvelopePolicy()
        if pol.require_signature and issuer_key is None:
            raise ValueError("issuer_key is required when policy requires signatures")
        return cls(header=header, policy=pol, issuer_key=issuer_key)

    @property
    def state(self) -> EnvelopeState:
        return self._state

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def header(self) -> EnvelopeHeader:
        return self._header

    @property
    def interim_commitments(self) -> List[InterimCommitment]:
        return list(self._interim_commitments)

    def ingest(self, token_record: Dict[str, Any]) -> "EnvelopeSession":
        """Add one token's evidence to the envelope.

        Args:
            token_record: Dict with at least ``token_index`` (int).
                Any additional data is hashed into the cumulative digest.

        Returns:
            self, for chaining.

        Raises:
            ValueError: If the session is sealed, expired, or token order violated.
        """
        if self._state != EnvelopeState.OPEN:
            raise ValueError(f"cannot ingest: session is {self._state.value}")
        if (time.time() - self._header.opened_at) > self._policy.ttl_seconds:
            self._state = EnvelopeState.EXPIRED
            raise ValueError("session expired")
        if self._token_count >= self._policy.max_token_count:
            raise ValueError(f"max token count {self._policy.max_token_count} exceeded")

        idx = int(token_record["token_index"])
        if self._policy.enforce_canonical_order and idx != self._last_token_index + 1:
            raise ValueError(
                f"non-contiguous token index: expected {self._last_token_index + 1}, got {idx}"
            )

        # Hash the token record in a deterministic way
        canonical = json.dumps(
            {k: v for k, v in sorted(token_record.items())},
            sort_keys=True, separators=(",", ":"), default=str,
        )
        self._hasher.update(canonical.encode("utf-8"))
        self._last_token_index = idx
        self._token_count += 1

        # Auto-commit check
        self._auto_commit_counter += 1
        if self._auto_commit_counter >= self._policy.commit_interval:
            self._auto_commit_counter = 0
            self._make_interim_commitment()

        return self

    def interim_commitment(self) -> InterimCommitment:
        """Manually create an interim commitment checkpoint.

        Returns:
            The ``InterimCommitment`` just created.

        Raises:
            ValueError: If session is not open or max commits exceeded.
        """
        if self._state != EnvelopeState.OPEN:
            raise ValueError(f"cannot commit: session is {self._state.value}")
        if len(self._interim_commitments) >= self._policy.max_interim_commits:
            raise ValueError("max interim commits exceeded")
        return self._make_interim_commitment()

    def _make_interim_commitment(self) -> InterimCommitment:
        ic = InterimCommitment(
            token_index=self._token_count - 1 if self._token_count > 0 else 0,
            cumulative_hash=self._hasher.hexdigest(),
            token_count=self._token_count,
            timestamp=time.time(),
            nonce=hashlib.sha256(
                f"{self._header.session_id}:{self._token_count}".encode()
            ).hexdigest()[:16],
        )
        self._interim_commitments.append(ic)
        return ic

    def seal(self, final_artifact_bytes: bytes | None = None) -> SealedEnvelope:
        """Seal the envelope, producing an immutable ``SealedEnvelope``.

        Args:
            final_artifact_bytes: The complete compact evidence artifact bytes.
                If provided, its SHA-256 is bound into the sealed envelope.

        Returns:
            A ``SealedEnvelope`` that can be verified post-hoc.

        Raises:
            ValueError: If the session is already sealed or expired.
        """
        if self._state == EnvelopeState.SEALED:
            raise ValueError("session already sealed")
        if self._state == EnvelopeState.EXPIRED:
            raise ValueError("session expired, cannot seal")

        # Make a final commitment if we have tokens and haven't auto-committed
        if self._token_count > 0:
            self._make_interim_commitment()

        artifact_hash = ""
        if final_artifact_bytes is not None:
            artifact_hash = hashlib.sha256(final_artifact_bytes).hexdigest()

        sealed = SealedEnvelope(
            header=self._header,
            final_cumulative_hash=self._hasher.hexdigest(),
            sealed_at=time.time(),
            total_tokens=self._token_count,
            interim_commitments=list(self._interim_commitments),
            artifact_content_hash=artifact_hash,
            seal_nonce=hashlib.sha256(
                f"seal:{self._header.session_id}:{time.time()}".encode()
            ).hexdigest()[:16],
        )

        # Sign if key is available
        if self._issuer_key is not None:
            sealed.signature = hmac.new(
                self._issuer_key,
                sealed.to_canonical_json().encode("utf-8"),
                hashlib.sha256,
            ).digest()

        self._state = EnvelopeState.SEALED
        return sealed


def verify_envelope(
    sealed: SealedEnvelope,
    *,
    issuer_key: bytes | None = None,
    artifact_bytes: bytes | None = None,
    tolerance_seconds: float = 5.0,
) -> Dict[str, Any]:
    """Verify a sealed envelope's integrity.

    Checks:
    1. Signature verification (if key provided)
    2. Artifact content hash (if artifact_bytes provided)
    3. Interim commitment chain integrity
    4. Final cumulative hash consistency
    5. Temporal consistency (seal time >= last commit time)

    Returns:
        ``{'valid': bool, 'errors': list[str]}``
    """
    errors: List[str] = []

    # 1. Signature
    if issuer_key is not None:
        if not sealed.signature:
            errors.append("envelope has no signature")
        else:
            expected = hmac.new(
                issuer_key,
                sealed.to_canonical_json().encode("utf-8"),
                hashlib.sha256,
            ).digest()
            if not hmac.compare_digest(sealed.signature, expected):
                errors.append("envelope signature invalid")

    # 2. Artifact content hash
    if artifact_bytes is not None:
        actual = hashlib.sha256(artifact_bytes).hexdigest()
        if sealed.artifact_content_hash and actual != sealed.artifact_content_hash:
            errors.append("artifact content hash mismatch")

    # 3. Interim commitment chain — verify monotonically increasing
    prev_idx = -1
    prev_hash = hashlib.sha256(sealed.header.to_canonical_json().encode("utf-8")).hexdigest()
    for ic in sealed.interim_commitments:
        if ic.token_index <= prev_idx:
            errors.append(f"interim commitment at token {ic.token_index} is not monotonic")
        # The cumulative_hash should include the header hash at the start,
        # so we can't fully re-verify without replaying all tokens.
        # But we can check internal consistency: each commitment's hash
        # should differ from the previous one if tokens were added.
        if ic.cumulative_hash == prev_hash and ic.token_count > 0:
            errors.append(f"interim commitment at token {ic.token_index} has unchanged cumulative hash")
        prev_idx = ic.token_index
        prev_hash = ic.cumulative_hash

    # 4. Final cumulative hash consistency with last interim commitment
    if sealed.interim_commitments:
        last_ic = sealed.interim_commitments[-1]
        if sealed.final_cumulative_hash != last_ic.cumulative_hash:
            # This is OK if tokens were added after the last interim commitment
            # but not OK if no tokens were added (the seal() method always
            # creates a final commitment, so they should match)
            if sealed.total_tokens == last_ic.token_count:
                errors.append("final cumulative hash does not match last interim commitment")

    # 5. Temporal consistency
    for ic in sealed.interim_commitments:
        if ic.timestamp > sealed.sealed_at + tolerance_seconds:
            errors.append(f"interim commitment at token {ic.token_index} is after seal time")
    if sealed.sealed_at < sealed.header.opened_at:
        errors.append("seal time is before session open time")

    return {"valid": len(errors) == 0, "errors": errors}


def verify_interim_commitment(
    sealed: SealedEnvelope,
    commitment_index: int,
    token_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Verify a specific interim commitment by replaying token records up to that point.

    This is the full verification path: replay the first N tokens, compute
    the cumulative hash, and compare against the stored commitment.

    Args:
        sealed: The sealed envelope.
        commitment_index: Index into ``sealed.interim_commitments``.
        token_records: The token records to replay.

    Returns:
        ``{'valid': bool, 'errors': list[str]}``
    """
    errors: List[str] = []

    if commitment_index < 0 or commitment_index >= len(sealed.interim_commitments):
        errors.append(f"commitment index {commitment_index} out of range")
        return {"valid": False, "errors": errors}

    ic = sealed.interim_commitments[commitment_index]
    if len(token_records) < ic.token_count:
        errors.append(
            f"insufficient token records: have {len(token_records)}, "
            f"need {ic.token_count} for commitment at index {commitment_index}"
        )
        return {"valid": False, "errors": errors}

    # Replay: hash header + first N token records
    hasher = hashlib.sha256()
    hasher.update(sealed.header.to_canonical_json().encode("utf-8"))
    for record in token_records[:ic.token_count]:
        canonical = json.dumps(
            {k: v for k, v in sorted(record.items())},
            sort_keys=True, separators=(",", ":"), default=str,
        )
        hasher.update(canonical.encode("utf-8"))

    actual_hash = hasher.hexdigest()
    if actual_hash != ic.cumulative_hash:
        errors.append(
            f"replayed hash {actual_hash[:16]}... does not match "
            f"commitment hash {ic.cumulative_hash[:16]}..."
        )

    return {"valid": len(errors) == 0, "errors": errors}
