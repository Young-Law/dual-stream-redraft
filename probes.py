from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import json
import math

import numpy as np

from .frame import ConceptScore

# The paper treats probe packs as evolving artifacts (Section 4.2).
# This module supports simple linear probes over hidden states:
#   score = sigmoid(w·h + b)
# Returned as sparse ConceptScore list filtered by threshold.


@dataclass(frozen=True)
class LinearProbe:
    concept_id: int
    layer: int  # which hidden-state layer to read from (0..L); -1 means last
    weight: List[float]
    bias: float = 0.0
    threshold: float = 0.5
    label: Optional[str] = None
    notes: Optional[str] = None

    def apply(self, h: np.ndarray) -> float:
        w = np.asarray(self.weight, dtype=np.float32)
        if w.shape[0] != h.shape[0]:
            raise ValueError(f"Probe dim mismatch: w={w.shape[0]} h={h.shape[0]}")
        z = float(np.dot(w, h) + self.bias)
        return 1.0 / (1.0 + math.exp(-z))


@dataclass
class ProbePack:
    probes: List[LinearProbe]
    vocab: Dict[int, str]

    @staticmethod
    def from_json(path: str) -> "ProbePack":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        probes = [LinearProbe(**p) for p in data.get("probes", [])]
        vocab = {int(k): str(v) for k, v in data.get("vocab", {}).items()}
        return ProbePack(probes=probes, vocab=vocab)


def run_probes(
    hidden_states: List[np.ndarray],
    pack: ProbePack,
    *,
    max_concepts: int = 32,
) -> List[ConceptScore]:
    """
    hidden_states: list of arrays [layer][hidden_dim] for the current token
    """
    hits: List[Tuple[int, float]] = []
    for p in pack.probes:
        layer_idx = p.layer
        if layer_idx == -1:
            layer_idx = len(hidden_states) - 1
        if layer_idx < 0 or layer_idx >= len(hidden_states):
            continue
        score = p.apply(hidden_states[layer_idx])
        if score >= p.threshold:
            hits.append((p.concept_id, score))

    # sort by score desc, keep sparse
    hits.sort(key=lambda x: x[1], reverse=True)
    hits = hits[:max_concepts]
    return [ConceptScore(concept_id=cid, score=float(score)) for cid, score in hits]
