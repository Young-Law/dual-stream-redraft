"""DSA v2.10 Retention Receipt Pipeline (§7).

Orchestrates the full end-to-end retention assurance pipeline:

1. **Issue** a ``RetentionRequirement`` for an artifact
2. **Store** the artifact via a ``StorageBackend``
3. **Validate** the stored artifact meets the requirement
4. **Issue** a ``RetentionReceipt`` chained to the requirement
5. **Possession** challenges to verify the artifact is still held
6. **Restore** verification after storage transforms

This module ties together ``retention_lifecycle``, ``storage_validator``,
``challenge``, ``migration``, and the ``VerifierWorkCertificate``.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .retention_lifecycle import (
    RetentionRequirement as LcRetentionRequirement,
    RetentionReceipt as LcRetentionReceipt,
    issue_retention_requirement,
    verify_retention_requirement,
    issue_retention_receipt,
    verify_retention_receipt,
    verify_receipt_chain,
)
from .storage_validator import (
    StorageBackend,
    LocalFilesystemBackend,
    StorageValidationResult,
    validate_persisted_artifact,
)
from .challenge import (
    PossessionChallenge,
    ChallengeResponse,
    issue_possession_challenge,
    respond_to_challenge,
    verify_possession_challenge,
)
from .migration import (
    AllowedTransform,
    verify_after_transform,
    MigrationVerificationResult,
)
from .retention import (
    RetentionReceipt as SpecRetentionReceipt,
    RetentionRequirement as SpecRetentionRequirement,
    compute_evidence_budget_summary,
    assert_retention_floor,
)


class PipelineStep(str, Enum):
    REQUIREMENT_ISSUED = "requirement_issued"
    ARTIFACT_STORED = "artifact_stored"
    STORAGE_VALIDATED = "storage_validated"
    RECEIPT_ISSUED = "receipt_issued"
    CHALLENGE_VERIFIED = "challenge_verified"
    RESTORE_VERIFIED = "restore_verified"
    CHAIN_VERIFIED = "chain_verified"


class PipelineStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class PipelineStepResult:
    """Result of a single pipeline step."""
    step: PipelineStep
    status: PipelineStatus
    timestamp: float = field(default_factory=time.time)
    detail: str = ""
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in (PipelineStatus.COMPLETED,)


@dataclass
class RetentionPipelineResult:
    """Complete result of the retention assurance pipeline."""
    artifact_id: str
    overall_status: PipelineStatus
    steps: List[PipelineStepResult] = field(default_factory=list)
    requirement: Optional[LcRetentionRequirement] = None
    receipt: Optional[LcRetentionReceipt] = None
    storage_result: Optional[StorageValidationResult] = None
    challenge_result: Optional[Dict[str, Any]] = None
    restore_result: Optional[MigrationVerificationResult] = None
    chain_result: Optional[Dict[str, Any]] = None
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        return (self.completed_at - self.started_at) if self.completed_at else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "overall_status": self.overall_status.value,
            "steps": [
                {"step": s.step.value, "status": s.status.value, "detail": s.detail, "errors": s.errors}
                for s in self.steps
            ],
            "elapsed_seconds": self.elapsed_seconds,
            "requirement_hash": self.requirement.hash().hex() if self.requirement else None,
            "receipt_hash": self.receipt.hash().hex() if self.receipt else None,
        }


class RetentionPipeline:
    """End-to-end retention assurance pipeline.

    Orchestrates the full lifecycle from requirement issuance through
    storage, validation, receipt creation, and possession challenges.

    Usage::

        pipeline = RetentionPipeline(
            storage_backend=LocalFilesystemBackend("/tmp/artifacts"),
            issuer_key=b"issuer-secret",
            validator_key=b"validator-secret",
            challenger_key=b"challenger-secret",
        )
        result = pipeline.run_full(
            artifact_id="run-001",
            artifact_bytes=evidence_data,
            profile_name="DSA-CI-Lite",
            issuer_id="governance-1",
            validator_id="storage-validator-1",
        )
        assert result.overall_status == PipelineStatus.COMPLETED
    """

    def __init__(
        self,
        storage_backend: Optional[StorageBackend] = None,
        issuer_key: bytes = b"",
        validator_key: bytes = b"",
        challenger_key: bytes = b"",
    ) -> None:
        self.storage = storage_backend or LocalFilesystemBackend("/tmp/dsa-artifacts")
        self.issuer_key = issuer_key
        self.validator_key = validator_key
        self.challenger_key = challenger_key

    def _step(
        self, step: PipelineStep, status: PipelineStatus, detail: str = "", errors: Optional[List[str]] = None
    ) -> PipelineStepResult:
        return PipelineStepResult(
            step=step, status=status, detail=detail, errors=errors or []
        )

    def run_full(
        self,
        *,
        artifact_id: str,
        artifact_bytes: bytes,
        profile_name: str = "DSA-CI-Lite",
        issuer_id: str = "",
        validator_id: str = "",
        min_retention_days: int = 90,
        max_artifact_bytes: int = 10_000_000,
        expires_at: Optional[float] = None,
        storage_class: str = "standard",
        run_possession_challenge: bool = True,
        verify_chain: bool = True,
    ) -> RetentionPipelineResult:
        """Run the full retention assurance pipeline.

        Steps:
        1. Issue RetentionRequirement
        2. Store artifact
        3. Validate stored artifact
        4. Issue RetentionReceipt
        5. (optional) Run possession challenge
        6. (optional) Verify full chain

        Returns:
            RetentionPipelineResult with detailed step results.
        """
        result = RetentionPipelineResult(
            artifact_id=artifact_id,
            overall_status=PipelineStatus.IN_PROGRESS,
        )
        steps = result.steps

        # Step 1: Issue requirement
        try:
            req = issue_retention_requirement(
                artifact_id=artifact_id,
                profile_name=profile_name,
                issuer_id=issuer_id,
                issuer_key=self.issuer_key,
                min_retention_days=min_retention_days,
                max_artifact_bytes=max_artifact_bytes,
                expires_at=expires_at,
            )
            result.requirement = req
            steps.append(self._step(PipelineStep.REQUIREMENT_ISSUED, PipelineStatus.COMPLETED,
                                    f"Requirement issued for {artifact_id}, retains until {time.strftime('%Y-%m-%d', time.gmtime(req.issued_at + min_retention_days * 86400))}"))
        except Exception as exc:
            steps.append(self._step(PipelineStep.REQUIREMENT_ISSUED, PipelineStatus.FAILED, str(exc)))
            result.overall_status = PipelineStatus.FAILED
            result.completed_at = time.time()
            return result

        # Step 2: Store artifact
        try:
            stored_bytes = self.storage.store(artifact_id, artifact_bytes)
            steps.append(self._step(PipelineStep.ARTIFACT_STORED, PipelineStatus.COMPLETED,
                                    f"Stored {stored_bytes} bytes for {artifact_id}"))
        except Exception as exc:
            steps.append(self._step(PipelineStep.ARTIFACT_STORED, PipelineStatus.FAILED, str(exc)))
            result.overall_status = PipelineStatus.PARTIAL
            result.completed_at = time.time()
            return result

        # Step 3: Validate stored artifact
        content_hash = hashlib.sha256(artifact_bytes).digest()
        validation = validate_persisted_artifact(
            self.storage, artifact_id, len(artifact_bytes), content_hash
        )
        result.storage_result = validation
        if validation.valid:
            steps.append(self._step(PipelineStep.STORAGE_VALIDATED, PipelineStatus.COMPLETED,
                                    f"Hash and size verified for {artifact_id}"))
        else:
            steps.append(self._step(PipelineStep.STORAGE_VALIDATED, PipelineStatus.FAILED,
                                    validation.error, [validation.error]))
            result.overall_status = PipelineStatus.PARTIAL
            result.completed_at = time.time()
            return result

        # Step 4: Issue receipt
        try:
            receipt = issue_retention_receipt(
                requirement=req,
                artifact_bytes=artifact_bytes,
                storage_backend=type(self.storage).__name__,
                validator_id=validator_id,
                validator_key=self.validator_key,
            )
            result.receipt = receipt
            steps.append(self._step(PipelineStep.RECEIPT_ISSUED, PipelineStatus.COMPLETED,
                                    f"Receipt issued, chained to requirement hash {req.hash().hex()[:16]}..."))
        except Exception as exc:
            steps.append(self._step(PipelineStep.RECEIPT_ISSUED, PipelineStatus.FAILED, str(exc)))
            result.overall_status = PipelineStatus.PARTIAL
            result.completed_at = time.time()
            return result

        # Step 5: Possession challenge (optional)
        if run_possession_challenge and self.challenger_key:
            try:
                challenge = issue_possession_challenge(
                    artifact_id=artifact_id,
                    artifact_bytes=artifact_bytes,
                    challenger_key=self.challenger_key,
                )
                response = respond_to_challenge(
                    challenge=challenge,
                    artifact_bytes=artifact_bytes,
                    responder_id=validator_id,
                    responder_key=self.validator_key,
                )
                challenge_result = verify_possession_challenge(
                    challenge=challenge,
                    response=response,
                    challenger_key=self.challenger_key,
                )
                result.challenge_result = challenge_result
                if challenge_result["valid"]:
                    steps.append(self._step(PipelineStep.CHALLENGE_VERIFIED, PipelineStatus.COMPLETED,
                                            f"Possession verified for {artifact_id}"))
                else:
                    steps.append(self._step(PipelineStep.CHALLENGE_VERIFIED, PipelineStatus.FAILED,
                                            "; ".join(challenge_result["errors"])))
            except Exception as exc:
                steps.append(self._step(PipelineStep.CHALLENGE_VERIFIED, PipelineStatus.FAILED, str(exc)))

        # Step 6: Verify full chain (optional)
        if verify_chain and result.requirement and result.receipt:
            chain = verify_receipt_chain(
                requirement=result.requirement,
                receipt=result.receipt,
                artifact_bytes=artifact_bytes,
                issuer_key=self.issuer_key,
                validator_key=self.validator_key,
            )
            result.chain_result = chain
            if chain["valid"]:
                steps.append(self._step(PipelineStep.CHAIN_VERIFIED, PipelineStatus.COMPLETED,
                                        "Full chain verified: requirement → receipt → content"))
            else:
                steps.append(self._step(PipelineStep.CHAIN_VERIFIED, PipelineStatus.FAILED,
                                        "; ".join(chain["errors"])))

        # Determine overall status
        failed = sum(1 for s in steps if s.status == PipelineStatus.FAILED)
        if failed == 0:
            result.overall_status = PipelineStatus.COMPLETED
        elif any(s.step in (PipelineStep.ARTIFACT_STORED, PipelineStep.STORAGE_VALIDATED, PipelineStep.RECEIPT_ISSUED)
                  and s.status == PipelineStatus.FAILED for s in steps):
            result.overall_status = PipelineStatus.FAILED
        else:
            result.overall_status = PipelineStatus.PARTIAL

        result.completed_at = time.time()
        return result

    def verify_restored_artifact(
        self,
        *,
        original_bytes: bytes,
        restored_bytes: bytes,
        transform: AllowedTransform = AllowedTransform.TRANSFORM_REPLICATE,
        retention_floor_bytes: int = 0,
    ) -> MigrationVerificationResult:
        """Verify an artifact after a storage transform (restore path)."""
        result = verify_after_transform(
            original_bytes=original_bytes,
            transformed_bytes=restored_bytes,
            transform=transform,
            retention_floor_bytes=retention_floor_bytes,
        )
        return result

    def run_possession_audit(
        self,
        *,
        artifact_id: str,
        artifact_bytes: bytes,
        num_challenges: int = 3,
    ) -> Dict[str, Any]:
        """Run multiple possession challenges to audit artifact retention.

        Issues ``num_challenges`` challenges with random byte ranges
        and verifies the holder can still prove possession.

        Returns:
            Dict with 'all_passed', 'results', 'summary'.
        """
        if not self.challenger_key:
            return {"all_passed": False, "results": [], "summary": "no challenger key configured"}

        total_len = len(artifact_bytes)
        results = []
        all_passed = True

        for i in range(num_challenges):
            import random
            start = random.randint(0, max(0, total_len - 1))
            end = min(total_len, start + random.randint(64, min(4096, total_len - start + 1)))

            challenge = issue_possession_challenge(
                artifact_id=artifact_id,
                artifact_bytes=artifact_bytes,
                challenger_key=self.challenger_key,
                start_offset=start,
                end_offset=end,
            )
            response = respond_to_challenge(
                challenge=challenge,
                artifact_bytes=artifact_bytes,
                responder_id="audit-responder",
                responder_key=self.validator_key,
            )
            verify_result = verify_possession_challenge(
                challenge=challenge,
                response=response,
                challenger_key=self.challenger_key,
            )
            results.append({
                "challenge_index": i,
                "range": [start, end],
                "valid": verify_result["valid"],
                "errors": verify_result["errors"],
            })
            if not verify_result["valid"]:
                all_passed = False

        passed_count = sum(1 for r in results if r["valid"])
        return {
            "all_passed": all_passed,
            "results": results,
            "summary": f"{passed_count}/{num_challenges} challenges passed",
        }
