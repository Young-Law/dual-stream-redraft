"""Dual-Stream Architecture (DSA) software-only reference implementation.

This package is intentionally modular: low-level codecs (frames) can be imported without requiring
heavy ML dependencies. The generation wrapper requires torch + transformers.
"""

from .frame import MonologueFrameV1, AttnSummary, ConceptScore, TopKToken, encode_frame, decode_frame
from .render import render_monologue_text
from .audit import coherence_audit, CoherenceFinding

__all__ = [
    "MonologueFrameV1",
    "TopKToken",
    "AttnSummary",
    "ConceptScore",
    "encode_frame",
    "decode_frame",
    "render_monologue_text",
    "coherence_audit",
    "CoherenceFinding",
]

# Optional: generation wrapper (torch/transformers)
try:
    from .generator import DualStreamGenerator, GenerationConfig  # noqa: F401

    __all__ += ["DualStreamGenerator", "GenerationConfig"]
except Exception:
    # Allow importing codecs/render/audit without ML stack installed.
    pass


# Optional ARC-DSA sidecar modules
try:
    from .arc_frame import ArcDecisionFrameV1, ArcConceptScore, HypothesisScore  # noqa: F401
    from .arc_audit import ArcCoherenceFinding, arc_coherence_audit  # noqa: F401
    from .arc_metrics import coherence_score  # noqa: F401

    __all__ += [
        "ArcDecisionFrameV1",
        "ArcConceptScore",
        "HypothesisScore",
        "ArcCoherenceFinding",
        "arc_coherence_audit",
        "coherence_score",
    ]
except Exception:
    pass


# v2.10 Phase 2: envelopes, triggers, retention pipeline, retry
try:
    from .envelope import (  # noqa: F401
        EnvelopeSession,
        EnvelopePolicy,
        EnvelopeHeader,
        SealedEnvelope,
        InterimCommitment,
        verify_envelope,
        verify_interim_commitment,
    )
    from .triggers import (  # noqa: F401
        ThresholdTrigger,
        ThresholdOperator,
        SequenceTrigger,
        SequenceMode,
        CompositeTrigger,
        CompositeOperator,
        TriggerPipeline,
        TriggerResult,
        TriggerBase,
    )
    from .retention_manager import (  # noqa: F401
        RetentionPipeline,
        PipelineStatus,
        PipelineStep,
        RetentionPipelineResult,
    )
    from .verifier import (  # noqa: F401
        RetryPolicy,
        RetryAttempt,
        RetryResult,
        verify_with_retry,
    )

    __all__ += [
        "EnvelopeSession",
        "EnvelopePolicy",
        "EnvelopeHeader",
        "SealedEnvelope",
        "InterimCommitment",
        "verify_envelope",
        "verify_interim_commitment",
        "ThresholdTrigger",
        "ThresholdOperator",
        "SequenceTrigger",
        "SequenceMode",
        "CompositeTrigger",
        "CompositeOperator",
        "TriggerPipeline",
        "TriggerResult",
        "TriggerBase",
        "RetentionPipeline",
        "PipelineStatus",
        "PipelineStep",
        "RetentionPipelineResult",
        "RetryPolicy",
        "RetryAttempt",
        "RetryResult",
        "verify_with_retry",
    ]
except Exception:
    pass
