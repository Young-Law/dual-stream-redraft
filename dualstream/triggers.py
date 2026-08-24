"""DSA v2.10 Trigger Classes (§6.2).

This module provides the three remaining trigger classes that complement
the existing bitmask-based trigger system in ``compact_evidence.py``:

- ``ThresholdTrigger`` — fires when a numeric signal crosses a threshold
- ``SequenceTrigger`` — fires based on sequence-level context (rolling windows,
  consecutive counts, historical tension)
- ``CompositeTrigger`` — combines child triggers with AND/OR/NOT logic

These are evaluated *before* encoding to set the ``trigger_flags`` bitmask
on each token record. They integrate with the existing ``_apply_v33_triggers()``
pipeline via the ``evaluate()`` method.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from .compact_evidence import (
    TRIGGER_CANARY,
    TRIGGER_ESCALATION,
    TRIGGER_HISTORY,
    TRIGGER_RANK,
    TRIGGER_STOCHASTIC,
)


# Map of well-known signal names to their evaluation functions
SIGNAL_NAMES = {
    "entropy": lambda ctx: ctx.get("entropy", 0.0),
    "refusal_mass": lambda ctx: ctx.get("refusal_mass", 0.0),
    "risk_score": lambda ctx: ctx.get("risk_score", 0.0),
    "affirmation_mass": lambda ctx: ctx.get("affirmation_mass", 0.0),
    "concept_score_max": lambda ctx: max(ctx.get("ast_scores", {}).values(), default=0.0),
    "concept_score_sum": lambda ctx: sum(ctx.get("ast_scores", {}).values()),
    "window_suspicious_rate": lambda ctx: ctx.get("suspicious_window_rate", 0.0),
    "rolling_entropy_mean": lambda ctx: ctx.get("rolling_entropy_mean", 0.0),
    "adaptive_fraction": lambda ctx: ctx.get("adaptive_fraction", 0.0),
}


@dataclass(frozen=True)
class TriggerResult:
    """Result of evaluating a single trigger."""
    fired: bool
    trigger_flag: int  # bitmask to OR into trigger_flags
    reason: str = ""
    detail: Any = None


class TriggerBase(ABC):
    """Abstract base class for all DSA v2.10 triggers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable trigger name."""
        ...

    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> TriggerResult:
        """Evaluate the trigger against the current context.

        Args:
            context: A dict containing signal values for the current token/sequence.
                Standard keys include ``token_index``, ``entropy``, ``refusal_mass``,
                ``ast_scores``, ``risk_score``, ``chosen_rank``, ``topk_ids``, etc.

        Returns:
            A ``TriggerResult`` with ``fired``, the ``trigger_flag`` bitmask,
            and a human-readable reason.
        """
        ...

    def reset(self) -> None:
        """Reset any internal state (e.g., rolling windows, counters)."""
        pass


# ---------------------------------------------------------------------------
# ThresholdTrigger
# ---------------------------------------------------------------------------


class ThresholdOperator(str, Enum):
    GT = ">"  # signal > threshold
    GTE = ">="  # signal >= threshold
    LT = "<"  # signal < threshold
    LTE = "<="  # signal <= threshold
    EQ = "=="  # signal == threshold
    NEQ = "!="  # signal != threshold


@dataclass
class ThresholdTrigger(TriggerBase):
    """Fires when a numeric signal crosses a threshold.

    This is the primary per-token trigger class. It evaluates a named signal
    (or a custom extractor) against a threshold using a comparison operator.

    Example usage::

        entropy_trigger = ThresholdTrigger(
            name="entropy_exceeds_4",
            signal="entropy",
            operator=ThresholdOperator.GTE,
            threshold=4.0,
            trigger_flag=TRIGGER_HISTORY,
            reason="Entropy {signal} {value:.3f} >= {threshold}",
        )
        result = entropy_trigger.evaluate({"entropy": 4.5, "token_index": 42})
        # result.fired == True
        # result.trigger_flag == TRIGGER_HISTORY (0x04)
    """
    name: str = ""
    signal: str = "entropy"  # key in SIGNAL_NAMES, or "custom"
    operator: ThresholdOperator = ThresholdOperator.GTE
    threshold: float = 0.0
    trigger_flag: int = TRIGGER_HISTORY
    reason_template: str = "{name}: {signal} {value:.4f} {operator} {threshold}"
    extractor: Optional[Callable[[Dict[str, Any]], float]] = None
    enabled: bool = True

    def evaluate(self, context: Dict[str, Any]) -> TriggerResult:
        if not self.enabled:
            return TriggerResult(fired=False, trigger_flag=0, reason="disabled")

        # Extract the signal value
        if self.extractor is not None:
            value = float(self.extractor(context))
        elif self.signal in SIGNAL_NAMES:
            value = float(SIGNAL_NAMES[self.signal](context))
        else:
            # Try to get directly from context
            value = float(context.get(self.signal, 0.0))

        # Evaluate comparison
        ops = {
            ThresholdOperator.GT: lambda v, t: v > t,
            ThresholdOperator.GTE: lambda v, t: v >= t,
            ThresholdOperator.LT: lambda v, t: v < t,
            ThresholdOperator.LTE: lambda v, t: v <= t,
            ThresholdOperator.EQ: lambda v, t: v == t,
            ThresholdOperator.NEQ: lambda v, t: v != t,
        }
        fired = ops[self.operator](value, self.threshold)

        if fired:
            reason = self.reason_template.format(
                name=self.name,
                signal=self.signal,
                value=value,
                operator=self.operator.value,
                threshold=self.threshold,
            )
        else:
            reason = ""

        return TriggerResult(
            fired=fired,
            trigger_flag=self.trigger_flag if fired else 0,
            reason=reason,
            detail={"signal": self.signal, "value": value, "threshold": self.threshold},
        )


# ---------------------------------------------------------------------------
# SequenceTrigger
# ---------------------------------------------------------------------------


class SequenceMode(str, Enum):
    CONSECUTIVE = "consecutive"  # N consecutive tokens meeting condition
    ROLLING_WINDOW = "rolling_window"  # Rate within last W tokens
    CUMULATIVE = "cumulative"  # Total count since last reset
    HISTORICAL_TENSION = "historical_tension"  # Based on TensionMap evaluation
    EVALUATION_CANARY = "evaluation_canary"  # Canary evaluation mode


@dataclass
class RollingWindow:
    """A fixed-size rolling window for tracking recent values."""
    capacity: int = 100
    _values: List[float] = field(default_factory=list, repr=False)

    def push(self, value: float) -> None:
        self._values.append(value)
        if len(self._values) > self.capacity:
            self._values.pop(0)

    @property
    def size(self) -> int:
        return len(self._values)

    def rate(self, predicate: Callable[[float], bool]) -> float:
        if not self._values:
            return 0.0
        return sum(1 for v in self._values if predicate(v)) / len(self._values)

    def mean(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0

    def max(self) -> float:
        return max(self._values) if self._values else 0.0

    def clear(self) -> None:
        self._values.clear()


@dataclass
class SequenceTrigger(TriggerBase):
    """Fires based on sequence-level patterns over multiple tokens.

    Supports multiple modes:

    - **consecutive**: Fires when N consecutive tokens meet a condition.
    - **rolling_window**: Fires when the rate within the last W tokens exceeds a threshold.
    - **cumulative**: Fires when the total count since last reset exceeds a threshold.
    - **historical_tension**: Wraps a TensionMap's ``evaluate_triggers()``.
    - **evaluation_canary**: Fires for every token when canary evaluation mode is active.

    Example usage::

        # Fire HISTORY when 5 consecutive tokens have entropy > 3.0
        entropy_streak = SequenceTrigger(
            name="entropy_streak_5",
            mode=SequenceMode.CONSECUTIVE,
            signal="entropy",
            condition=lambda v: v > 3.0,
            count_threshold=5,
            trigger_flag=TRIGGER_HISTORY,
        )

        # Fire ESCALATION when 20% of last 100 tokens are suspicious
        suspicious_rate = SequenceTrigger(
            name="high_suspicious_rate",
            mode=SequenceMode.ROLLING_WINDOW,
            signal="risk_score",
            condition=lambda v: v > 0.5,
            window_size=100,
            rate_threshold=0.2,
            trigger_flag=TRIGGER_ESCALATION,
        )
    """
    name: str = ""
    mode: SequenceMode = SequenceMode.CONSECUTIVE
    signal: str = "entropy"
    condition: Callable[[float], bool] = lambda v: v > 0.0
    count_threshold: int = 3
    window_size: int = 100
    rate_threshold: float = 0.2
    trigger_flag: int = TRIGGER_HISTORY
    tension_map: Any = None  # TensionMap instance for HISTORICAL_TENSION mode
    enabled: bool = True
    # Internal state
    _consecutive_count: int = field(default=0, init=False, repr=False)
    _cumulative_count: int = field(default=0, init=False, repr=False)
    _window: RollingWindow = field(default_factory=lambda: RollingWindow(100), init=False, repr=False)

    def __post_init__(self):
        if self.mode == SequenceMode.ROLLING_WINDOW:
            object.__setattr__(self, "_window", RollingWindow(self.window_size))

    def evaluate(self, context: Dict[str, Any]) -> TriggerResult:
        if not self.enabled:
            return TriggerResult(fired=False, trigger_flag=0, reason="disabled")

        if self.mode == SequenceMode.HISTORICAL_TENSION:
            return self._evaluate_historical_tension(context)
        if self.mode == SequenceMode.EVALUATION_CANARY:
            return self._evaluate_canary(context)

        # Extract signal value
        if self.signal in SIGNAL_NAMES:
            value = float(SIGNAL_NAMES[self.signal](context))
        else:
            value = float(context.get(self.signal, 0.0))

        meets_condition = self.condition(value)

        if self.mode == SequenceMode.CONSECUTIVE:
            return self._evaluate_consecutive(meets_condition, value, context)
        elif self.mode == SequenceMode.ROLLING_WINDOW:
            return self._evaluate_rolling_window(meets_condition, value, context)
        elif self.mode == SequenceMode.CUMULATIVE:
            return self._evaluate_cumulative(meets_condition, value, context)

        return TriggerResult(fired=False, trigger_flag=0, reason="unknown mode")

    def _evaluate_consecutive(
        self, meets_condition: bool, value: float, context: Dict[str, Any]
    ) -> TriggerResult:
        if meets_condition:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0

        if self._consecutive_count >= self.count_threshold:
            reason = (
                f"{self.name}: {self.signal} met condition for "
                f"{self._consecutive_count} consecutive tokens (threshold {self.count_threshold})"
            )
            return TriggerResult(fired=True, trigger_flag=self.trigger_flag, reason=reason)

        return TriggerResult(fired=False, trigger_flag=0)

    def _evaluate_rolling_window(
        self, meets_condition: bool, value: float, context: Dict[str, Any]
    ) -> TriggerResult:
        self._window.push(1.0 if meets_condition else 0.0)
        rate = self._window.rate(lambda v: v > 0.5)

        if self._window.size >= 10 and rate >= self.rate_threshold:
            reason = (
                f"{self.name}: {self.signal} condition rate {rate:.2%} "
                f"within last {self._window.size} tokens (threshold {self.rate_threshold:.2%})"
            )
            return TriggerResult(fired=True, trigger_flag=self.trigger_flag, reason=reason)

        return TriggerResult(fired=False, trigger_flag=0)

    def _evaluate_cumulative(
        self, meets_condition: bool, value: float, context: Dict[str, Any]
    ) -> TriggerResult:
        if meets_condition:
            self._cumulative_count += 1

        if self._cumulative_count >= self.count_threshold:
            reason = (
                f"{self.name}: {self.signal} met condition "
                f"{self._cumulative_count} times cumulatively (threshold {self.count_threshold})"
            )
            return TriggerResult(fired=True, trigger_flag=self.trigger_flag, reason=reason)

        return TriggerResult(fired=False, trigger_flag=0)

    def _evaluate_historical_tension(self, context: Dict[str, Any]) -> TriggerResult:
        if self.tension_map is None:
            return TriggerResult(fired=False, trigger_flag=0, reason="no tension map")
        triggered = self.tension_map.evaluate_triggers(context)
        if triggered:
            return TriggerResult(
                fired=True,
                trigger_flag=TRIGGER_HISTORY,
                reason=f"{self.name}: tension map rule matched for context",
            )
        return TriggerResult(fired=False, trigger_flag=0)

    def _evaluate_canary(self, context: Dict[str, Any]) -> TriggerResult:
        canary_eval = context.get("canary_eval", False)
        if canary_eval:
            return TriggerResult(
                fired=True,
                trigger_flag=TRIGGER_CANARY,
                reason=f"{self.name}: canary evaluation mode active",
            )
        return TriggerResult(fired=False, trigger_flag=0)

    def reset(self) -> None:
        self._consecutive_count = 0
        self._cumulative_count = 0
        self._window.clear()


# ---------------------------------------------------------------------------
# CompositeTrigger
# ---------------------------------------------------------------------------


class CompositeOperator(str, Enum):
    AND = "and"  # all children must fire
    OR = "or"  # any child must fire
    NOT = "not"  # inverts a single child
    NAND = "nand"  # not all children fire (at least one must not)
    MAJORITY = "majority"  # more than half of children fire


@dataclass
class CompositeTrigger(TriggerBase):
    """Combines multiple child triggers with boolean logic.

    Supports AND, OR, NOT, NAND, and MAJORITY operators. The resulting
    ``trigger_flag`` is the OR of all fired children's flags.

    Example usage::

        entropy_t = ThresholdTrigger(
            name="high_entropy", signal="entropy",
            operator=ThresholdOperator.GTE, threshold=4.0,
        )
        risk_t = ThresholdTrigger(
            name="high_risk", signal="risk_score",
            operator=ThresholdOperator.GTE, threshold=0.7,
            trigger_flag=TRIGGER_ESCALATION,
        )
        # Fire ESCALATION only when BOTH entropy and risk are high
        composite = CompositeTrigger(
            name="high_entropy_and_risk",
            operator=CompositeOperator.AND,
            children=[entropy_t, risk_t],
            trigger_flag=TRIGGER_ESCALATION,
        )
    """
    name: str = ""
    operator: CompositeOperator = CompositeOperator.OR
    children: List[TriggerBase] = field(default_factory=list)
    trigger_flag: int = TRIGGER_ESCALATION  # default when composite fires
    enabled: bool = True

    def evaluate(self, context: Dict[str, Any]) -> TriggerResult:
        if not self.enabled:
            return TriggerResult(fired=False, trigger_flag=0, reason="disabled")

        if not self.children:
            return TriggerResult(fired=False, trigger_flag=0, reason="no children")

        results = [child.evaluate(context) for child in self.children]
        fired_flags = set()
        fired_reasons: List[str] = []
        fired_details: List[Dict[str, Any]] = []
        children_fired_count = 0

        for r in results:
            if r.fired:
                children_fired_count += 1
                fired_flags.add(r.trigger_flag)
                if r.reason:
                    fired_reasons.append(r.reason)
                if r.detail:
                    fired_details.append(r.detail)

        total_count = len(results)

        # Apply composite operator
        ops = {
            CompositeOperator.AND: children_fired_count == total_count and total_count > 0,
            CompositeOperator.OR: children_fired_count > 0,
            CompositeOperator.NOT: children_fired_count == 0 and total_count == 1,
            CompositeOperator.NAND: children_fired_count < total_count,
            CompositeOperator.MAJORITY: children_fired_count > total_count / 2,
        }
        fired = ops[self.operator]

        if fired:
            combined_flag = self.trigger_flag
            # Also OR in all children's flags for full visibility
            for f in fired_flags:
                combined_flag |= f

            reason = (
                f"{self.name} ({self.operator.value}): "
                + "; ".join(fired_reasons) if fired_reasons else self.name
            )
            return TriggerResult(
                fired=True,
                trigger_flag=combined_flag,
                reason=reason,
                detail={"children_fired": children_fired_count, "children_total": total_count},
            )

        return TriggerResult(fired=False, trigger_flag=0)

    def reset(self) -> None:
        for child in self.children:
            child.reset()


# ---------------------------------------------------------------------------
# TriggerPipeline — orchestrates multiple triggers per token
# ---------------------------------------------------------------------------


@dataclass
class TriggerPipeline:
    """Evaluates a sequence of triggers and aggregates results.

    The pipeline processes triggers in order and ORs together all fired
    flags. It maintains stateful triggers (SequenceTrigger) across tokens.

    Example usage::

        pipeline = TriggerPipeline(triggers=[
            ThresholdTrigger("entropy", "entropy", ThresholdOperator.GTE, 4.0),
            SequenceTrigger("streak", mode=SequenceMode.CONSECUTIVE, count_threshold=5),
            CompositeTrigger("complex", operator=CompositeOperator.AND, children=[...]),
        ])
        for context in token_contexts:
            result = pipeline.evaluate(context)
            if result.fired:
                flags |= result.trigger_flag
    """
    triggers: List[TriggerBase] = field(default_factory=list)

    def evaluate(self, context: Dict[str, Any]) -> TriggerResult:
        """Evaluate all triggers against the context."""
        combined_flag = 0
        reasons: List[str] = []
        details: List[Dict[str, Any]] = []
        any_fired = False

        for trigger in self.triggers:
            result = trigger.evaluate(context)
            if result.fired:
                any_fired = True
                combined_flag |= result.trigger_flag
                if result.reason:
                    reasons.append(result.reason)
                if result.detail:
                    details.append(result.detail)

        return TriggerResult(
            fired=any_fired,
            trigger_flag=combined_flag,
            reason="; ".join(reasons) if reasons else "",
            detail=details if details else None,
        )

    def reset(self) -> None:
        for trigger in self.triggers:
            trigger.reset()

    def add_trigger(self, trigger: TriggerBase) -> "TriggerPipeline":
        self.triggers.append(trigger)
        return self
