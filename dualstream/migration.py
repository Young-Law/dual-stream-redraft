"""DSA v2.10 Migration/Restore Policy — verify artifacts after storage transform."""

import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AllowedTransform(Enum):
    TRANSFORM_NONE = "none"
    TRANSFORM_COMPRESS = "compress"        # Lossless compression (zlib, gzip)
    TRANSFORM_REPLICATE = "replicate"        # Geographic replication
    TRANSFORM_REENCODE = "reencode"          # Wire format version upgrade
    TRANSFORM_PARTITION = "partition"        # Sharding / partitioning
    TRANSFORM_ENCRYPT = "encrypt"            # At-rest encryption


@dataclass
class MigrationVerificationResult:
    """Result of verifying an artifact after storage transform."""
    valid: bool
    transform: str
    original_hash: bytes
    transformed_hash: bytes
    content_intact: bool
    retention_floor_met: bool
    verified_at: float
    errors: list = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def verify_after_transform(
    original_bytes: bytes,
    transformed_bytes: bytes,
    transform: AllowedTransform,
    retention_floor_bytes: int = 0,
    allowed_transforms: Optional[list[AllowedTransform]] = None,
) -> MigrationVerificationResult:
    """Verify restored canonical bytes. Only replication and zlib/gzip are implemented.

    Encryption, partitioning and reencoding require explicit reconstruction
    adapters; unsupported transforms fail closed instead of inferring integrity.
    """
    errors = []
    original_hash = hashlib.sha256(original_bytes).digest()
    transformed_hash = hashlib.sha256(transformed_bytes).digest()
    restored = transformed_bytes
    if allowed_transforms is not None and transform not in allowed_transforms:
        errors.append("Transform not in allowed list")
    if transform == AllowedTransform.TRANSFORM_COMPRESS:
        import zlib
        try:
            decoder = zlib.decompressobj(wbits=47)
            # Bound decompression by trusted original size, including malformed bombs.
            restored = decoder.decompress(transformed_bytes, len(original_bytes) + 1)
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError("Incomplete, oversized, or trailing compressed stream")
        except (zlib.error, ValueError) as exc:
            errors.append(f"Cannot reconstruct compressed artifact: {exc}")
            restored = b""
    elif transform not in (AllowedTransform.TRANSFORM_NONE, AllowedTransform.TRANSFORM_REPLICATE):
        errors.append("Transform reconstruction adapter not implemented")
        restored = b""
    content_intact = restored == original_bytes
    if not content_intact:
        errors.append("Restored artifact differs from original")
    try:
        from .compact_evidence import decode_compact_sequence
        decoded = decode_compact_sequence(restored)
        manifest = decoded.get("manifest")
        canonical_floor = manifest.minimum_reconstructable_bytes if manifest else len(restored)
    except Exception as exc:
        errors.append(f"Restored artifact does not fully decode: {exc}")
        content_intact = False
        canonical_floor = len(original_bytes)
    retention_floor_met = retention_floor_bytes >= 0 and len(restored) >= max(retention_floor_bytes, canonical_floor)
    if not retention_floor_met:
        errors.append("Restored artifact below retention floor")
    return MigrationVerificationResult(
        valid=not errors, transform=transform.value, original_hash=original_hash,
        transformed_hash=transformed_hash, content_intact=content_intact,
        retention_floor_met=retention_floor_met, verified_at=time.time(), errors=errors)
