from __future__ import annotations

from typing import Iterable, List, Optional, Dict, Any, Tuple
import math

from .frame import MonologueFrameV1
from .vocab import concept_label

def _fmt_prob(p: float) -> str:
    # stable short formatting, similar to the paper's example
    return f"{p:.2f}".rstrip("0").rstrip(".") if p < 1 else "1.0"

def render_monologue_text(
    frames: Iterable[MonologueFrameV1],
    *,
    tokenizer_decode: Optional[callable] = None,
    topk_label: str = "LOGITS_TOP",
) -> str:
    """
    Render evidence frames into a human-readable monologue string suitable for log review
    and promptfoo assertions.

    This output is *not* the monologue stream itself; it is a renderer over structured frames.
    """
    lines: List[str] = []
    for fr in frames:
        # Concepts first, to match Appendix A's illustrative example
        for c in fr.concepts:
            lines.append(f"[{concept_label(c.concept_id)} score={_fmt_prob(c.score)}]")

        # Logits top-K
        items: List[str] = []
        for tk in fr.topk:
            tok = None
            if tokenizer_decode is not None:
                try:
                    tok = tokenizer_decode([tk.token_id])
                except Exception:
                    tok = None
            tok = tok if tok is not None else str(tk.token_id)
            # sanitize newlines to keep one line
            tok = tok.replace("\n", "\\n")
            items.append(f'("{tok}",{_fmt_prob(tk.prob)})')
        lines.append(f"[{topk_label}{len(fr.topk)}: " + " ".join(items) + "]")

        # Optional attention summaries (compact)
        if fr.attn:
            # render at most 8 for readability
            for a in fr.attn[:8]:
                lines.append(
                    f"[ATTN layer={a.layer} head={a.head} top_token_idx={a.top_token_idx} weight={_fmt_prob(a.weight)}]"
                )

    return "\n".join(lines).strip() + ("\n" if lines else "")
