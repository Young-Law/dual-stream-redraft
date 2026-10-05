"""Create a deterministic H4-C design-freeze manifest.

Run this only after the preregistration, confirmatory scenario set, runner, evaluator,
analysis code, model revision, audit rate, treatment version, and sample-size planning
decision are final. The manifest contains hashes and identifiers only; it never contains
the protected audit key.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(
    *,
    git_commit: str,
    model_id: str,
    model_revision: str,
    treatment_version: str,
    audit_rate_ppm: int,
    number_of_pairs: int,
    planning_p10: float,
    planning_p01: float,
    alpha_one_sided: float,
    target_power: float,
    preregistration: Path,
    scenario_set: Path,
    evaluator: Path,
    runner: Path,
    analysis: Path,
) -> dict:
    if len(git_commit.strip()) < 7:
        raise ValueError("git_commit must identify the frozen repository revision")
    if not model_id.strip() or not model_revision.strip():
        raise ValueError("model_id and model_revision must be nonempty")
    if not treatment_version.strip():
        raise ValueError("treatment_version must be nonempty")
    if not 0 < audit_rate_ppm < 1_000_000:
        raise ValueError("audit_rate_ppm must be between 1 and 999999")
    if number_of_pairs < 1:
        raise ValueError("number_of_pairs must be positive")
    if not 0.0 <= planning_p01 < planning_p10 <= 1.0:
        raise ValueError("planning probabilities must satisfy 0 <= p01 < p10 <= 1")
    if planning_p10 + planning_p01 > 1.0:
        raise ValueError("planning_p10 + planning_p01 must not exceed 1")
    if not 0.0 < alpha_one_sided < 1.0:
        raise ValueError("alpha_one_sided must be between 0 and 1")
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be between 0 and 1")

    files = {
        "preregistration": preregistration,
        "scenario_set": scenario_set,
        "evaluator": evaluator,
        "runner": runner,
        "analysis": analysis,
    }
    for name, path in files.items():
        if not path.is_file():
            raise ValueError(f"{name} file does not exist: {path}")

    return {
        "schema_version": "h4c-design-freeze-v2",
        "status": "frozen-before-confirmatory-data",
        "git_commit": git_commit.strip(),
        "model_id": model_id.strip(),
        "model_revision": model_revision.strip(),
        "treatment_version": treatment_version.strip(),
        "audit_rate_ppm": audit_rate_ppm,
        "number_of_confirmatory_pairs": number_of_pairs,
        "sample_size_planning": {
            "alternative": "C1 SAER > C2 SAER",
            "p10_c1_success_c2_failure": planning_p10,
            "p01_c1_failure_c2_success": planning_p01,
            "alpha_one_sided": alpha_one_sided,
            "target_power": target_power,
        },
        "files": {
            name: {
                "path": path.as_posix(),
                "sha256": _sha256(path),
            }
            for name, path in files.items()
        },
        "protected_key_in_manifest": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an H4-C design-freeze manifest")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--treatment-version", required=True)
    parser.add_argument("--audit-rate-ppm", type=int, required=True)
    parser.add_argument("--number-of-pairs", type=int, required=True)
    parser.add_argument("--planning-p10", type=float, required=True)
    parser.add_argument("--planning-p01", type=float, required=True)
    parser.add_argument("--alpha-one-sided", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.90)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("docs/h4c_model_in_loop_preregistration.md"),
    )
    parser.add_argument("--scenario-set", type=Path, required=True)
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path("experiments/h4c_model_in_loop.py"),
    )
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("experiments/h4c_paired_analysis.py"),
    )
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    try:
        manifest = build_manifest(
            git_commit=args.git_commit,
            model_id=args.model,
            model_revision=args.model_revision,
            treatment_version=args.treatment_version,
            audit_rate_ppm=args.audit_rate_ppm,
            number_of_pairs=args.number_of_pairs,
            planning_p10=args.planning_p10,
            planning_p01=args.planning_p01,
            alpha_one_sided=args.alpha_one_sided,
            target_power=args.target_power,
            preregistration=args.preregistration,
            scenario_set=args.scenario_set,
            evaluator=args.evaluator,
            runner=args.runner,
            analysis=args.analysis,
        )
    except ValueError as exc:
        parser.error(str(exc))

    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
