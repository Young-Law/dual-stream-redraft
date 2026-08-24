"""DSA v2.10 Retention Lifecycle — Requirement issuance, Receipt creation, Chain verification."""

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetentionRequirement:
    """A signed retention requirement issued by a governance authority."""
    artifact_id: str
    profile_name: str
    issuer_id: str
    min_retention_days: int
    max_artifact_bytes: int
    issued_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    nonce: str = ""
    signature: bytes = b""

    def to_canonical_json(self) -> str:
        """Serialize to canonical JSON for signing and hashing."""
        payload = {
            "artifact_id": self.artifact_id,
            "profile_name": self.profile_name,
            "issuer_id": self.issuer_id,
            "min_retention_days": self.min_retention_days,
            "max_artifact_bytes": self.max_artifact_bytes,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "nonce": self.nonce,
        }
        return json.dumps(payload, sort_keys=True, separators=(',', ':'))

    def hash(self) -> bytes:
        """SHA-256 hash of the canonical JSON representation."""
        return hashlib.sha256(self.to_canonical_json().encode()).digest()


@dataclass 
class RetentionReceipt:
    """A receipt proving an artifact was stored according to a requirement."""
    requirement_hash: bytes  # SHA-256 of the RetentionRequirement
    artifact_content_hash: bytes  # SHA-256 of actual artifact bytes
    storage_backend: str  # e.g. "local-fs", "s3"
    stored_bytes: int
    stored_at: float = field(default_factory=time.time)
    validator_id: str = ""
    nonce: str = ""
    signature: bytes = b""

    def to_canonical_json(self) -> str:
        payload = {
            "requirement_hash": self.requirement_hash.hex(),
            "artifact_content_hash": self.artifact_content_hash.hex(),
            "storage_backend": self.storage_backend,
            "stored_bytes": self.stored_bytes,
            "stored_at": self.stored_at,
            "validator_id": self.validator_id,
            "nonce": self.nonce,
        }
        return json.dumps(payload, sort_keys=True, separators=(',', ':'))

    def hash(self) -> bytes:
        return hashlib.sha256(self.to_canonical_json().encode()).digest()


def issue_retention_requirement(
    artifact_id: str,
    profile_name: str,
    issuer_id: str,
    issuer_key: bytes,
    min_retention_days: int = 90,
    max_artifact_bytes: int = 10_000_000,
    expires_at: Optional[float] = None,
) -> RetentionRequirement:
    """Create and sign a new RetentionRequirement."""
    import secrets
    req = RetentionRequirement(
        artifact_id=artifact_id,
        profile_name=profile_name,
        issuer_id=issuer_id,
        min_retention_days=min_retention_days,
        max_artifact_bytes=max_artifact_bytes,
        expires_at=expires_at,
        nonce=secrets.token_hex(16),
    )
    canonical = req.to_canonical_json()
    req.signature = hmac.new(issuer_key, canonical.encode(), hashlib.sha256).digest()
    return req


def verify_retention_requirement(req: RetentionRequirement, issuer_key: bytes) -> bool:
    """Verify the HMAC signature on a RetentionRequirement."""
    expected = hmac.new(issuer_key, req.to_canonical_json().encode(), hashlib.sha256).digest()
    return hmac.compare_digest(req.signature, expected)


def issue_retention_receipt(
    requirement: RetentionRequirement,
    artifact_bytes: bytes,
    storage_backend: str,
    validator_id: str,
    validator_key: bytes,
    stored_bytes: Optional[int] = None,
) -> RetentionReceipt:
    """Create and sign a RetentionReceipt chained to a RetentionRequirement."""
    import secrets
    content_hash = hashlib.sha256(artifact_bytes).digest()
    receipt = RetentionReceipt(
        requirement_hash=requirement.hash(),
        artifact_content_hash=content_hash,
        storage_backend=storage_backend,
        stored_bytes=stored_bytes or len(artifact_bytes),
        validator_id=validator_id,
        nonce=secrets.token_hex(16),
    )
    canonical = receipt.to_canonical_json()
    receipt.signature = hmac.new(validator_key, canonical.encode(), hashlib.sha256).digest()
    return receipt


def verify_retention_receipt(receipt: RetentionReceipt, validator_key: bytes) -> bool:
    """Verify the HMAC signature on a RetentionReceipt."""
    expected = hmac.new(validator_key, receipt.to_canonical_json().encode(), hashlib.sha256).digest()
    return hmac.compare_digest(receipt.signature, expected)


def verify_receipt_chain(
    requirement: RetentionRequirement,
    receipt: RetentionReceipt,
    artifact_bytes: bytes,
    issuer_key: bytes,
    validator_key: bytes,
) -> dict:
    """
    Full chain verification: requirement signature → receipt signature → hash chain → content match.
    Returns a dict with 'valid': bool and 'errors': list of strings.
    """
    errors = []
    
    # 1. Verify requirement signature
    if not verify_retention_requirement(requirement, issuer_key):
        errors.append('Requirement signature invalid')
    
    # 2. Verify receipt signature
    if not verify_retention_receipt(receipt, validator_key):
        errors.append('Receipt signature invalid')
    
    # 3. Verify hash chain: receipt.requirement_hash == requirement.hash()
    if receipt.requirement_hash != requirement.hash():
        errors.append('Receipt not chained to requirement')
    
    # 4. Verify artifact content hash
    actual_content_hash = hashlib.sha256(artifact_bytes).digest()
    if receipt.artifact_content_hash != actual_content_hash:
        errors.append('Artifact content hash mismatch')
    
    # 5. Verify stored bytes match
    if receipt.stored_bytes != len(artifact_bytes):
        errors.append(f'Stored bytes mismatch: receipt says {receipt.stored_bytes}, actual {len(artifact_bytes)}')
    
    # 6. Check expiry if set
    if requirement.expires_at and time.time() > requirement.expires_at:
        errors.append('Requirement has expired')
    
    return {'valid': len(errors) == 0, 'errors': errors}
