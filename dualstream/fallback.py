from __future__ import annotations

from dataclasses import dataclass

SAFE_FALLBACK_TEXT = "Sorry, I can’t help with that request. I can help with a safe alternative or a benign version."


@dataclass(frozen=True)
class FallbackDecision:
    action: str
    reason: str
    retry_count: int
    max_retries: int
    fallback_text: str


class FallbackRouter:
    def __init__(self, *, max_retries: int = 1, strategy: str = "canned_refusal"):
        self.max_retries = max(0, int(max_retries))
        self.strategy = strategy

    def decide(self, *, red_state: bool, retry_count: int) -> FallbackDecision:
        if not red_state:
            return FallbackDecision("none", "pass", retry_count, self.max_retries, "")
        if retry_count >= self.max_retries:
            action = self.strategy if self.strategy in {"canned_refusal", "safe_override", "abort"} else "canned_refusal"
            text = SAFE_FALLBACK_TEXT if action != "abort" else ""
            return FallbackDecision(action, "retry_budget_exceeded", retry_count, self.max_retries, text)
        return FallbackDecision("none", "retry_allowed", retry_count, self.max_retries, "")
