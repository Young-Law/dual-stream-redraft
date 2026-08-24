"""DSA v2.10 Possession Challenge Protocol — prove artifact still held."""

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PossessionChallenge:
    """A challenge issued to prove an artifact is still held."""
    challenge_id: str
    artifact_id: str
    # Token range to prove possession of
    start_offset: int
    end_offset: int
    # HMAC of the expected bytes (computed by challenger who knows the artifact)
    expected_hash: bytes
    nonce: str
    issued_at: float = field(default_factory=time.time)
    expires_at: float = 0.0  # default 5 min
    challenge_key: bytes = b""
    signature: bytes = b""

    def __post_init__(self):
        if self.expires_at == 0.0:
            self.expires_at = self.issued_at + 300  # 5 minutes

    def to_canonical_json(self) -> str:
        import json
        payload = {
            "challenge_id": self.challenge_id,
            "artifact_id": self.artifact_id,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "expected_hash": self.expected_hash.hex(),
            "nonce": self.nonce,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }
        return json.dumps(payload, sort_keys=True, separators=(',', ':'))


@dataclass
class ChallengeResponse:
    """Response to a possession challenge."""
    challenge_id: str
    responder_id: str
    provided_hash: bytes
    responded_at: float = field(default_factory=time.time)
    nonce: str = ""
    signature: bytes = b""

    def to_canonical_json(self) -> str:
        import json
        payload = {
            "challenge_id": self.challenge_id,
            "responder_id": self.responder_id,
            "provided_hash": self.provided_hash.hex(),
            "responded_at": self.responded_at,
            "nonce": self.nonce,
        }
        return json.dumps(payload, sort_keys=True, separators=(',', ':'))


def issue_possession_challenge(
    artifact_id: str,
    artifact_bytes: bytes,
    challenger_key: bytes,
    start_offset: int = 0,
    end_offset: Optional[int] = None,
    ttl_seconds: int = 300,
) -> PossessionChallenge:
    """
    Issue a possession challenge.
    
    Selects a random byte range from the artifact, computes the expected hash,
    signs it, and returns the challenge.
    """
    total_len = len(artifact_bytes)
    if end_offset is None:
        end_offset = total_len
    
    # Clamp to valid range
    start_offset = max(0, min(start_offset, total_len))
    end_offset = max(start_offset, min(end_offset, total_len))
    
    # Compute expected hash of the selected range
    selected_bytes = artifact_bytes[start_offset:end_offset]
    expected_hash = hashlib.sha256(selected_bytes).digest()
    
    challenge = PossessionChallenge(
        challenge_id=secrets.token_hex(16),
        artifact_id=artifact_id,
        start_offset=start_offset,
        end_offset=end_offset,
        expected_hash=expected_hash,
        nonce=secrets.token_hex(16),
        challenge_key=challenger_key,
        expires_at=time.time() + ttl_seconds,
    )
    
    # Sign the challenge
    canonical = challenge.to_canonical_json()
    challenge.signature = hmac.new(challenger_key, canonical.encode(), hashlib.sha256).digest()
    return challenge


def verify_possession_challenge(
    challenge: PossessionChallenge,
    response: ChallengeResponse,
    challenger_key: bytes,
) -> dict:
    """
    Verify a possession challenge response.
    
    Returns {'valid': bool, 'errors': list[str]}
    """
    errors = []
    
    # 1. Check not expired
    if time.time() > challenge.expires_at:
        errors.append('Challenge expired')
    
    # 2. Verify challenge signature
    expected_sig = hmac.new(
        challenger_key, challenge.to_canonical_json().encode(), hashlib.sha256
    ).digest()
    if not hmac.compare_digest(challenge.signature, expected_sig):
        errors.append('Challenge signature invalid')
    
    # 3. Check challenge ID matches
    if challenge.challenge_id != response.challenge_id:
        errors.append('Challenge ID mismatch')
    
    # 4. Verify response hash matches expected
    if not hmac.compare_digest(response.provided_hash, challenge.expected_hash):
        errors.append('Provided hash does not match expected range hash')
    
    return {'valid': len(errors) == 0, 'errors': errors}


def respond_to_challenge(
    challenge: PossessionChallenge,
    artifact_bytes: bytes,
    responder_id: str,
    responder_key: bytes,
) -> ChallengeResponse:
    """
    Respond to a possession challenge by computing the hash of the requested range.
    """
    selected = artifact_bytes[challenge.start_offset:challenge.end_offset]
    provided_hash = hashlib.sha256(selected).digest()
    
    response = ChallengeResponse(
        challenge_id=challenge.challenge_id,
        responder_id=responder_id,
        provided_hash=provided_hash,
        nonce=secrets.token_hex(16),
    )
    
    # Sign response
    canonical = response.to_canonical_json()
    response.signature = hmac.new(responder_key, canonical.encode(), hashlib.sha256).digest()
    return response
