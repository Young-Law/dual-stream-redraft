from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import hashlib

# Appendix B (v2.2 redraft) defines integrity primitives (frame CRC + sequence running hash).
# In the software-only path, this module provides a minimal running-hash accumulator.

DEFAULT_HASH = "sha256"


@dataclass
class RunningHash:
    algo: str = DEFAULT_HASH
    _h: "hashlib._Hash" = None  # type: ignore

    def __post_init__(self) -> None:
        self._h = hashlib.new(self.algo)

    def update(self, frame_bytes: bytes) -> None:
        self._h.update(frame_bytes)

    def digest_hex(self) -> str:
        return self._h.hexdigest()

    def digest_bytes(self) -> bytes:
        return self._h.digest()


def verify_running_hash(frames: list[bytes], expected_hex: str, algo: str = DEFAULT_HASH) -> bool:
    h = hashlib.new(algo)
    for fb in frames:
        h.update(fb)
    return h.hexdigest().lower() == expected_hex.lower()
