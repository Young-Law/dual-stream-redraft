"""Local research possession checks using authenticated range preimages.

The holder must return actual bytes, not echo a public digest. This deliberately
uses bandwidth proportional to the challenged range; it is not a succinct proof
of retrievability, a future-retention guarantee, or independent attestation.
"""
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PossessionChallenge:
    challenge_id: str
    artifact_id: str
    start_offset: int
    end_offset: int
    expected_hash: bytes
    nonce: str
    issued_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    signature: bytes = b""

    def __post_init__(self):
        if self.expires_at == 0.0:
            self.expires_at = self.issued_at + 300

    def to_canonical_json(self) -> str:
        return json.dumps({k: (v.hex() if isinstance(v, bytes) else v)
                           for k, v in vars(self).items() if k != 'signature'},
                          sort_keys=True, separators=(',', ':'))


@dataclass
class ChallengeResponse:
    challenge_id: str
    responder_id: str
    provided_hash: bytes
    responded_at: float = field(default_factory=time.time)
    nonce: str = ""
    signature: bytes = b""
    provided_bytes: bytes = b""

    def to_canonical_json(self) -> str:
        return json.dumps({k: (v.hex() if isinstance(v, bytes) else v)
                           for k, v in vars(self).items() if k != 'signature'},
                          sort_keys=True, separators=(',', ':'))


def issue_possession_challenge(artifact_id: str, artifact_bytes: bytes,
                               challenger_key: bytes, start_offset: int = 0,
                               end_offset: Optional[int] = None,
                               ttl_seconds: int = 300) -> PossessionChallenge:
    if not challenger_key or not artifact_id or ttl_seconds <= 0:
        raise ValueError('Nonempty key, artifact ID and positive TTL required')
    if end_offset is None:
        end_offset = len(artifact_bytes)
    if not 0 <= start_offset < end_offset <= len(artifact_bytes):
        raise ValueError('Challenge must select a nonempty valid byte range')
    challenge = PossessionChallenge(
        secrets.token_hex(16), artifact_id, start_offset, end_offset,
        hashlib.sha256(artifact_bytes[start_offset:end_offset]).digest(),
        secrets.token_hex(16), expires_at=time.time() + ttl_seconds)
    challenge.signature = hmac.digest(challenger_key, challenge.to_canonical_json().encode(), 'sha256')
    return challenge


def verify_possession_challenge(challenge: PossessionChallenge,
                                response: ChallengeResponse,
                                challenger_key: bytes,
                                responder_key: bytes = b"",
                                expected_responder_id: Optional[str] = None) -> dict:
    errors = []
    now = time.time()
    if not challenger_key or not responder_key:
        errors.append('Both trusted verification keys are required')
    if not challenge.issued_at <= response.responded_at <= now <= challenge.expires_at:
        errors.append('Challenge expired or response timestamp invalid')
    if not hmac.compare_digest(challenge.signature, hmac.digest(
            challenger_key, challenge.to_canonical_json().encode(), 'sha256')):
        errors.append('Challenge signature invalid')
    if not hmac.compare_digest(response.signature, hmac.digest(
            responder_key, response.to_canonical_json().encode(), 'sha256')):
        errors.append('Response signature invalid')
    if response.challenge_id != challenge.challenge_id or response.nonce != challenge.nonce:
        errors.append('Challenge ID or nonce mismatch')
    if not response.responder_id or (expected_responder_id is not None and response.responder_id != expected_responder_id):
        errors.append('Responder identity mismatch')
    if challenge.start_offset < 0 or challenge.end_offset <= challenge.start_offset:
        errors.append('Invalid challenge range')
    if len(response.provided_bytes) != challenge.end_offset - challenge.start_offset:
        errors.append('Provided byte range length mismatch')
    actual_hash = hashlib.sha256(response.provided_bytes).digest()
    if not hmac.compare_digest(actual_hash, challenge.expected_hash):
        errors.append('Provided bytes do not match trusted range commitment')
    if not hmac.compare_digest(actual_hash, response.provided_hash):
        errors.append('Provided hash does not match provided bytes')
    return {'valid': not errors, 'errors': errors}


def respond_to_challenge(challenge: PossessionChallenge, artifact_bytes: bytes,
                         responder_id: str, responder_key: bytes) -> ChallengeResponse:
    if not responder_key or not responder_id:
        raise ValueError('Nonempty responder key and identity required')
    if not 0 <= challenge.start_offset < challenge.end_offset <= len(artifact_bytes):
        raise ValueError('Stored artifact cannot satisfy challenge range')
    selected = artifact_bytes[challenge.start_offset:challenge.end_offset]
    response = ChallengeResponse(challenge.challenge_id, responder_id,
                                 hashlib.sha256(selected).digest(), nonce=challenge.nonce,
                                 provided_bytes=selected)
    response.signature = hmac.digest(responder_key, response.to_canonical_json().encode(), 'sha256')
    return response
