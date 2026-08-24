"""DSA v2.10 Storage Validator — Abstract storage backend + persisted-byte validation."""

import hashlib
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class StorageValidationResult:
    """Result of validating persisted artifact storage."""
    valid: bool
    backend: str
    artifact_id: str
    stored_bytes: int
    expected_bytes: int
    content_hash: bytes
    expected_content_hash: bytes
    hash_matches: bool
    size_matches: bool
    validated_at: float = field(default_factory=time.time)
    error: str = ""


class StorageBackend(ABC):
    """Abstract storage backend for DSA artifacts."""

    @abstractmethod
    def store(self, artifact_id: str, data: bytes) -> int:
        """Store artifact bytes, return number of bytes stored."""
        ...

    @abstractmethod
    def retrieve(self, artifact_id: str) -> Optional[bytes]:
        """Retrieve artifact bytes, or None if not found."""
        ...

    @abstractmethod
    def exists(self, artifact_id: str) -> bool:
        """Check if artifact exists in storage."""
        ...

    @abstractmethod
    def delete(self, artifact_id: str) -> bool:
        """Delete artifact from storage."""
        ...

    @abstractmethod
    def size(self, artifact_id: str) -> Optional[int]:
        """Get size of stored artifact, or None if not found."""
        ...


class LocalFilesystemBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, artifact_id: str) -> Path:
        # Use SHA-256 of artifact_id as filename to avoid path traversal
        safe_name = hashlib.sha256(artifact_id.encode()).hexdigest()
        return self.base_dir / safe_name

    def store(self, artifact_id: str, data: bytes) -> int:
        path = self._path(artifact_id)
        path.write_bytes(data)
        return len(data)

    def retrieve(self, artifact_id: str) -> Optional[bytes]:
        path = self._path(artifact_id)
        if not path.exists():
            return None
        return path.read_bytes()

    def exists(self, artifact_id: str) -> bool:
        return self._path(artifact_id).exists()

    def delete(self, artifact_id: str) -> bool:
        path = self._path(artifact_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def size(self, artifact_id: str) -> Optional[int]:
        path = self._path(artifact_id)
        if not path.exists():
            return None
        return path.stat().st_size


def validate_persisted_artifact(
    backend: StorageBackend,
    artifact_id: str,
    expected_bytes: int,
    expected_content_hash: bytes,
) -> StorageValidationResult:
    """
    Validate that a persisted artifact matches expectations.

    Checks:
    1. Artifact exists in storage
    2. Stored byte count matches expected
    3. Content SHA-256 hash matches expected

    Returns a StorageValidationResult with detailed status.
    """
    # Check existence
    if not backend.exists(artifact_id):
        return StorageValidationResult(
            valid=False,
            backend=type(backend).__name__,
            artifact_id=artifact_id,
            stored_bytes=0,
            expected_bytes=expected_bytes,
            content_hash=b"",
            expected_content_hash=expected_content_hash,
            hash_matches=False,
            size_matches=False,
            error="Artifact not found in storage",
        )

    # Retrieve and validate
    data = backend.retrieve(artifact_id)
    if data is None:
        return StorageValidationResult(
            valid=False,
            backend=type(backend).__name__,
            artifact_id=artifact_id,
            stored_bytes=0,
            expected_bytes=expected_bytes,
            content_hash=b"",
            expected_content_hash=expected_content_hash,
            hash_matches=False,
            size_matches=False,
            error="Artifact exists but could not be retrieved",
        )

    stored_bytes = len(data)
    actual_hash = hashlib.sha256(data).digest()
    size_matches = stored_bytes == expected_bytes
    hash_matches = actual_hash == expected_content_hash

    errors = []
    if not size_matches:
        errors.append(f"Size mismatch: stored {stored_bytes}, expected {expected_bytes}")
    if not hash_matches:
        errors.append("Content hash mismatch")

    return StorageValidationResult(
        valid=len(errors) == 0,
        backend=type(backend).__name__,
        artifact_id=artifact_id,
        stored_bytes=stored_bytes,
        expected_bytes=expected_bytes,
        content_hash=actual_hash,
        expected_content_hash=expected_content_hash,
        hash_matches=hash_matches,
        size_matches=size_matches,
        error="; ".join(errors) if errors else "",
    )
