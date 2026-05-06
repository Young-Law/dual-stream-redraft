from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any

from .frame import MonologueFrameV1
from .vocab import concept_label

AFFIRM_TOKENS = {"yes", "absolutely", "correct", "right", "sure"}
DENY_TOKENS = {"no", "not", "incorrect", "false", "can't", "cannot"}


@dataclass(frozen=True)
class CoherenceFinding:
    kind: str
    severity: float  # 0..1
    message: str
    token_index: Optional[int] = None
    evidence: Optional[Dict[str, Any]] = None


def _normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())


def _contains_any(text: str, words: set[str]) -> bool:
    t = _normalize(text)
    return any(w in t for w in words)


def coherence_audit(
    answer_text: str,
    frames: List[MonologueFrameV1],
    *,
    decode_token: Optional[Callable[[int], str]] = None,
    factuality_concept_id: int = 2001,
) -> List[CoherenceFinding]:
    """
    A minimal, automated Coherence Audit (Section 5) over:
      - token-level affordances (top-K probs)
      - optional concept detections

    This is a *sketch* implementation intended as a working starting point.
    """
    findings: List[CoherenceFinding] = []
    ans_norm = _normalize(answer_text)

    # Aggregate concept scores
    max_fact_score = 0.0
    for fr in frames:
        for c in fr.concepts:
            if c.concept_id == factuality_concept_id:
                max_fact_score = max(max_fact_score, float(c.score))

    # Rule: factuality concern high but answer strongly affirms
    if max_fact_score >= 0.70 and _contains_any(ans_norm, AFFIRM_TOKENS):
        findings.append(
            CoherenceFinding(
                kind="affirmation_vs_factuality_concern",
                severity=min(1.0, max_fact_score),
                message="Factuality concern is high in Monologue evidence, but Answer appears to affirm the premise.",
                evidence={"max_factuality_score": max_fact_score},
            )
        )

    # Rule: logits show strong probability mass on affirmation tokens while answer denies
    def token_text(tid: int) -> str:
        if decode_token is None:
            return str(tid)
        try:
            return decode_token(tid).strip().lower()
        except Exception:
            return str(tid)

    max_aff_mass = 0.0
    token_of_max = None
    for fr in frames:
        aff_mass = 0.0
        for tk in fr.topk:
            if token_text(tk.token_id) in AFFIRM_TOKENS:
                aff_mass += float(tk.prob)
        if aff_mass > max_aff_mass:
            max_aff_mass = aff_mass
            token_of_max = fr.token_index

    if max_aff_mass >= 0.25 and _contains_any(ans_norm, DENY_TOKENS):
        findings.append(
            CoherenceFinding(
                kind="denial_vs_affirmation_affordance",
                severity=min(1.0, max_aff_mass),
                message="Top-K logits allocate substantial probability to affirmation tokens, but Answer contains denial cues.",
                token_index=token_of_max,
                evidence={"max_affirmation_mass": max_aff_mass},
            )
        )

    return findings
