"""Create a deterministic H4-C design-freeze manifest.

Run this only after the preregistration, confirmatory scenario set, runner, evaluator,
analysis code, model revision, and sample-size decision are final. The manifest contains
hashes and identifiers only; it never contains the protected audit key.
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
    number_of_pairs: int,
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
    if number_of_pairs < 1:
        raise ValueError("number_of_pairs must be positive")

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
        "schema_version": "h4c-design-freeze-v1",
        "status": "frozen-before-confirmatory-data",
        "git_commit": git_commit.strip(),
        "model_id": model_id.strip(),
        "model_revision": model_revision.strip(),
        "number_of_confirmatory_pairs": number_of_pairs,
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
    parser.add_argument("--number-of-pairs", type=int, required=True)
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
            number_of_pairs=args.number_of_pairs,
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
