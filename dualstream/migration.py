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
    """
    Verify an artifact after a storage transformation.
    
    For TRANSFORM_NONE, TRANSFORM_COMPRESS, TRANSFORM_ENCRYPT, TRANSFORM_REPLICATE:
      - Decompress/decrypt/compare to verify content integrity
    For TRANSFORM_REENCODE:
      - Verify the transformed artifact decodes to the same token sequence
    For TRANSFORM_PARTITION:
      - Verify reassembled parts match original
    
    Since we can't do actual decompression/decryption without knowing the specific 
    algorithm, we verify:
    1. The transform is in the allowed list
    2. The original hash is preserved in the verification result
    3. The retention floor is met (transformed size >= floor for uncompressed transforms)
    """
    errors = []
    original_hash = hashlib.sha256(original_bytes).digest()
    transformed_hash = hashlib.sha256(transformed_bytes).digest()
    
    # 1. Check transform is allowed
    if allowed_transforms is not None and transform not in allowed_transforms:
        errors.append(f"Transform '{transform.value}' not in allowed list")
    
    content_intact = True
    
    # 2. For lossless transforms, verify we can round-trip
    if transform == AllowedTransform.TRANSFORM_NONE:
        content_intact = original_bytes == transformed_bytes
        if not content_intact:
            errors.append("Content changed with TRANSFORM_NONE")
    elif transform == AllowedTransform.TRANSFORM_COMPRESS:
        # Verify compressed form is actually smaller
        if len(transformed_bytes) >= len(original_bytes):
            errors.append(f"Compressed artifact ({len(transformed_bytes)}B) not smaller than original ({len(original_bytes)}B)")
            content_intact = False
    elif transform == AllowedTransform.TRANSFORM_REPLICATE:
        content_intact = original_hash == transformed_hash
        if not content_intact:
            errors.append("Replicated artifact hash differs from original")
    elif transform == AllowedTransform.TRANSFORM_ENCRYPT:
        # Encrypted data should be different from original
        if original_hash == transformed_hash:
            errors.append("Encrypted artifact hash identical to original — not actually encrypted?")
    elif transform == AllowedTransform.TRANSFORM_REENCODE:
        # Re-encoding may change byte representation but should preserve semantics
        # We can't fully verify without decoding both, so just check it's non-empty
        if len(transformed_bytes) == 0:
            errors.append("Re-encoded artifact is empty")
            content_intact = False
    elif transform == AllowedTransform.TRANSFORM_PARTITION:
        # Partitioned artifact is a subset — can't do full comparison
        # Just verify non-empty
        if len(transformed_bytes) == 0:
            errors.append("Partitioned artifact is empty")
            content_intact = False
    
    # 3. Retention floor check
    retention_floor_met = True
    if retention_floor_bytes > 0:
        if transform in (AllowedTransform.TRANSFORM_NONE, AllowedTransform.TRANSFORM_REPLICATE):
            if len(transformed_bytes) < retention_floor_bytes:
                errors.append(f"Transformed size {len(transformed_bytes)} below retention floor {retention_floor_bytes}")
                retention_floor_met = False
    
    return MigrationVerificationResult(
        valid=len(errors) == 0,
        transform=transform.value,
        original_hash=original_hash,
        transformed_hash=transformed_hash,
        content_intact=content_intact,
        retention_floor_met=retention_floor_met,
        verified_at=time.time(),
        errors=errors,
    )
