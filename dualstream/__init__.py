"""Dual-Stream Architecture (DSA) software-only reference implementation.

This package is intentionally modular: low-level codecs (frames) can be imported without
requiring heavy ML dependencies. The generation wrapper targets local Ollama models.
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

from .generator import DualStreamGenerator, GenerationConfig  # noqa: F401

__all__ += ["DualStreamGenerator", "GenerationConfig"]
