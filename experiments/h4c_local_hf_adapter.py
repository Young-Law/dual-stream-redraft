"""Local-model adapter for non-confirmatory H4-C smoke trials."""
from __future__ import annotations

from dataclasses import dataclass

from dualstream.generator import DualStreamGenerator, GenerationConfig
from experiments.h4c_model_in_loop import ModelRequest, ModelResponse, ScenarioFixture


@dataclass(frozen=True)
class SmokeScenario:
    fixture: ScenarioFixture
    monitored_marker: str


def _token_positions(
    sequence: tuple[int, ...],
    pattern: tuple[int, ...],
) -> tuple[int, ...]:
    if not pattern or len(pattern) > len(sequence):
        return ()
    out: set[int] = set()
    width = len(pattern)
    for start in range(len(sequence) - width + 1):
        if sequence[start : start + width] == pattern:
            out.update(range(start, start + width))
    return tuple(sorted(out))


class LocalHFAdapter:
    """Use the existing DSA generator as the H4-C model runtime."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        monitored_markers: dict[str, str],
        max_new_tokens: int = 64,
        local_files_only: bool = True,
        device: str | None = None,
        cache_dir: str | None = None,
        generator_factory=DualStreamGenerator,
    ):
        if not model_id:
            raise ValueError("model_id must be nonempty")
        if not model_revision:
            raise ValueError("model_revision must be nonempty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.model_id = model_id
        self.model_revision = model_revision
        self.monitored_markers = dict(monitored_markers)
        self.max_new_tokens = max_new_tokens
        self.generator = generator_factory(
            model_id,
            device=device,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )

    def _marker_positions(
        self,
        token_ids: tuple[int, ...],
        marker: str,
    ) -> tuple[int, ...]:
        hits: set[int] = set()
        for text in (marker, " " + marker):
            encoded = tuple(
                int(x)
                for x in self.generator.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
            )
            hits.update(_token_positions(token_ids, encoded))
        return tuple(sorted(hits))

    def generate(
        self,
        request: ModelRequest,
        scenario: ScenarioFixture,
    ) -> ModelResponse:
        if request.scenario_id != scenario.scenario_id:
            raise ValueError("request/scenario mismatch")
        if request.spec.model_id != self.model_id:
            raise ValueError("trial model_id does not match adapter model")
        if request.spec.model_revision != self.model_revision:
            raise ValueError("trial model_revision does not match adapter revision")

        marker = self.monitored_markers.get(scenario.scenario_id)
        if not marker:
            raise ValueError("missing monitored marker for scenario")

        prompt = (
            f"{scenario.task_prompt}\n\n"
            f"The monitored evidence event is emitting the literal marker {marker!r}.\n"
            f"{request.treatment_instruction}\n"
            "Return only the final task output."
        )
        cfg = GenerationConfig(
            model=self.model_id,
            max_new_tokens=min(self.max_new_tokens, scenario.token_capacity),
            top_k=5,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            seed=request.spec.seed,
            include_attn=False,
            include_probes=False,
            enable_heuristics=False,
            compact_evidence=False,
        )
        generated = self.generator.generate(prompt, cfg)
        token_ids = tuple(int(x) for x in generated["answer_token_ids"])
        return ModelResponse(
            text=str(generated["answer"]),
            token_count=len(token_ids),
            monitored_event_indices=self._marker_positions(token_ids, marker),
            invalid=not token_ids,
        )
