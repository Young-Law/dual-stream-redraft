"""Create and validate a fail-closed H4-C design-freeze manifest.

The manifest binds every behavior-affecting H4-C input that must remain fixed before
confirmatory data collection. Protected audit-key material is never accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from experiments.h4c_power import exact_directional_mcnemar_power

MANIFEST_SCHEMA = "h4c-design-freeze-v3"
RUNTIME_SCHEMA = "h4c-confirmatory-runtime-v1"
SCENARIO_SCHEMA = "h4c-model-confirmatory-v1"

_CANONICAL_FILE_PATHS = {
    "preregistration": Path("docs/h4c_model_in_loop_preregistration.md"),
    "evaluator": Path("experiments/h4c_model_in_loop.py"),
    "adapter": Path("experiments/h4c_local_hf_adapter.py"),
    "runner": Path("experiments/h4c_local_model_confirmatory.py"),
    "analysis": Path("experiments/h4c_paired_analysis.py"),
    "power_helper": Path("experiments/h4c_power.py"),
    "generator_runtime": Path("dualstream/generator.py"),
    "schedule_implementation": Path("dualstream/compact_evidence.py"),
    "freeze_validator": Path("scripts/h4c_freeze_manifest.py"),
}

_REQUIRED_FILE_ROLES = {
    "preregistration",
    "scenario_set",
    "runtime_config",
    "evaluator",
    "adapter",
    "runner",
    "analysis",
    "power_helper",
    "generator_runtime",
    "schedule_implementation",
    "freeze_validator",
}
_FORBIDDEN_RUNTIME_KEYS = {
    "protected_key",
    "protected_key_hex",
    "secret_key",
    "secret_key_hex",
    "h4c_protected_key_hex",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_git_commit(ref: str, repo_root: Path = Path(".")) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("unable to resolve declared repository commit") from exc
    commit_sha = completed.stdout.strip().lower()
    if len(commit_sha) != 40 or any(ch not in "0123456789abcdef" for ch in commit_sha):
        raise ValueError("declared repository commit is not a full SHA-1")
    return commit_sha


def _sha256_at_commit(repo_root: Path, commit_sha: str, path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "show", f"{commit_sha}:{path.as_posix()}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"frozen file is absent from declared repository commit: {path.as_posix()}"
        ) from exc
    return hashlib.sha256(completed.stdout).hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _contains_forbidden_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_RUNTIME_KEYS:
                return True
            if _contains_forbidden_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def validate_runtime_config(data: dict) -> dict:
    if _contains_forbidden_key(data):
        raise ValueError("runtime config must not contain protected key material")
    expected_fields = {
        "schema_version",
        "treatment_version",
        "seeds",
        "audit_rate_ppm",
        "audit_key_id",
        "key_commitment",
        "policy_version",
        "benchmark_id",
        "profile_id",
        "device",
        "decoding",
        "environment",
    }
    if set(data) != expected_fields:
        raise ValueError("runtime config fields do not match frozen schema")
    if data.get("schema_version") != RUNTIME_SCHEMA:
        raise ValueError("unsupported H4-C confirmatory runtime schema")

    treatment = str(data.get("treatment_version", "")).strip()
    if not treatment:
        raise ValueError("runtime treatment_version must be nonempty")

    seeds = data.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("runtime seeds must be a nonempty unique integer list")

    rate = data.get("audit_rate_ppm")
    if type(rate) is not int or not 0 < rate < 1_000_000:
        raise ValueError("runtime audit_rate_ppm must be between 1 and 999999")

    key_id = data.get("audit_key_id")
    if type(key_id) is not int or key_id < 0:
        raise ValueError("runtime audit_key_id must be a nonnegative integer")
    key_commitment = str(data.get("key_commitment", "")).lower()
    if (
        len(key_commitment) != 64
        or any(ch not in "0123456789abcdef" for ch in key_commitment)
    ):
        raise ValueError("runtime key_commitment must be a 64-character SHA-256 hex digest")

    for field in ("policy_version", "benchmark_id", "profile_id"):
        if not str(data.get(field, "")).strip():
            raise ValueError(f"runtime {field} must be nonempty")

    decoding = data.get("decoding")
    required_decoding = {
        "max_new_tokens",
        "top_k",
        "temperature",
        "top_p",
        "do_sample",
        "repetition_penalty",
        "no_repeat_ngram_size",
    }
    if not isinstance(decoding, dict) or set(decoding) != required_decoding:
        raise ValueError("runtime decoding fields do not match frozen schema")
    if type(decoding["max_new_tokens"]) is not int or decoding["max_new_tokens"] < 1:
        raise ValueError("runtime max_new_tokens must be positive")
    if type(decoding["top_k"]) is not int or decoding["top_k"] < 1:
        raise ValueError("runtime top_k must be positive")
    if (
        not isinstance(decoding["temperature"], (int, float))
        or isinstance(decoding["temperature"], bool)
        or decoding["temperature"] <= 0
    ):
        raise ValueError("runtime temperature must be positive")
    if (
        not isinstance(decoding["top_p"], (int, float))
        or isinstance(decoding["top_p"], bool)
        or not 0 < decoding["top_p"] <= 1
    ):
        raise ValueError("runtime top_p must be in (0, 1]")
    if type(decoding["do_sample"]) is not bool:
        raise ValueError("runtime do_sample must be boolean")
    if (
        not isinstance(decoding["repetition_penalty"], (int, float))
        or isinstance(decoding["repetition_penalty"], bool)
        or decoding["repetition_penalty"] <= 0
    ):
        raise ValueError("runtime repetition_penalty must be positive")
    if (
        type(decoding["no_repeat_ngram_size"]) is not int
        or decoding["no_repeat_ngram_size"] < 0
    ):
        raise ValueError("runtime no_repeat_ngram_size must be nonnegative")

    device = data.get("device")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("runtime device must be an explicit nonempty string")

    environment = data.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("runtime environment must be an object")
    if set(environment) != {
        "python_version",
        "platform_system",
        "platform_machine",
        "packages",
    }:
        raise ValueError("runtime environment fields do not match frozen schema")
    for field in ("python_version", "platform_system", "platform_machine"):
        if not str(environment.get(field, "")).strip():
            raise ValueError(f"runtime environment.{field} must be nonempty")
    packages = environment.get("packages")
    if not isinstance(packages, dict):
        raise ValueError("runtime environment.packages must be an object")
    for required in ("torch", "transformers", "tokenizers"):
        if not str(packages.get(required, "")).strip():
            raise ValueError(
                f"runtime environment.packages.{required} must be pinned"
            )
    return data


def _resolve_repo_file(repo_root: Path, path: Path, *, label: str) -> Path:
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} path must be repository-relative")
    resolved = repo_root / path
    if not resolved.is_file():
        raise ValueError(f"{label} file does not exist: {path}")
    return resolved


def _scenario_count(path: Path) -> int:
    data = _load_json_object(path, label="scenario_set")
    if data.get("schema_version") != SCENARIO_SCHEMA:
        raise ValueError("unsupported H4-C confirmatory scenario schema")
    scope = str(data.get("scope", ""))
    if "confirmatory" not in scope.casefold():
        raise ValueError("confirmatory scenario scope must be explicit")
    rows = data.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ValueError("confirmatory scenario set must be nonempty")
    ids = [str(row.get("scenario_id", "")).strip() for row in rows if isinstance(row, dict)]
    if len(ids) != len(rows) or any(not scenario_id for scenario_id in ids):
        raise ValueError("every confirmatory scenario must have a scenario_id")
    if len(set(ids)) != len(ids):
        raise ValueError("confirmatory scenario ids must be unique")
    return len(rows)


def _validate_canonical_execution_paths(file_paths: dict[str, Path]) -> None:
    for role, expected_path in _CANONICAL_FILE_PATHS.items():
        actual_path = file_paths.get(role)
        if actual_path != expected_path:
            raise ValueError(
                f"{role} must use canonical path {expected_path.as_posix()}"
            )


def build_manifest(
    *,
    git_commit: str,
    model_id: str,
    model_revision: str,
    number_of_pairs: int,
    planning_p10: float,
    planning_p01: float,
    alpha_one_sided: float,
    target_power: float,
    preregistration: Path,
    scenario_set: Path,
    runtime_config: Path,
    evaluator: Path,
    adapter: Path,
    runner: Path,
    analysis: Path,
    power_helper: Path,
    generator_runtime: Path,
    schedule_implementation: Path,
    freeze_validator: Path,
    repo_root: Path = Path("."),
) -> dict:
    resolved_git_commit = _resolve_git_commit(git_commit.strip(), Path("."))
    if not model_id.strip() or not model_revision.strip():
        raise ValueError("model_id and model_revision must be nonempty")
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

    file_paths = {
        "preregistration": preregistration,
        "scenario_set": scenario_set,
        "runtime_config": runtime_config,
        "evaluator": evaluator,
        "adapter": adapter,
        "runner": runner,
        "analysis": analysis,
        "power_helper": power_helper,
        "generator_runtime": generator_runtime,
        "schedule_implementation": schedule_implementation,
        "freeze_validator": freeze_validator,
    }
    _validate_canonical_execution_paths(file_paths)
    resolved = {
        name: _resolve_repo_file(repo_root, path, label=name)
        for name, path in file_paths.items()
    }
    for name, path in file_paths.items():
        working_hash = _sha256(resolved[name])
        committed_hash = _sha256_at_commit(Path("."), resolved_git_commit, path)
        if working_hash != committed_hash:
            raise ValueError(
                f"{name} does not match the declared repository commit"
            )

    runtime = validate_runtime_config(
        _load_json_object(resolved["runtime_config"], label="runtime_config")
    )
    scenarios = _scenario_count(resolved["scenario_set"])
    derived_pairs = scenarios * len(runtime["seeds"])
    if derived_pairs != number_of_pairs:
        raise ValueError(
            "number_of_pairs must equal confirmatory scenarios x frozen seeds"
        )

    achieved_power = exact_directional_mcnemar_power(
        number_of_pairs,
        p10=planning_p10,
        p01=planning_p01,
        alpha=alpha_one_sided,
    )
    if achieved_power < target_power:
        raise ValueError("frozen pair count is underpowered for declared target")

    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": "frozen-before-confirmatory-data",
        "git_commit": resolved_git_commit,
        "model_id": model_id.strip(),
        "model_revision": model_revision.strip(),
        "treatment_version": runtime["treatment_version"],
        "audit_rate_ppm": runtime["audit_rate_ppm"],
        "confirmatory_seeds": list(runtime["seeds"]),
        "scenario_schema": SCENARIO_SCHEMA,
        "runtime_schema": RUNTIME_SCHEMA,
        "number_of_confirmatory_pairs": number_of_pairs,
        "sample_size_planning": {
            "alternative": "C1 SAER > C2 SAER",
            "p10_c1_success_c2_failure": planning_p10,
            "p01_c1_failure_c2_success": planning_p01,
            "alpha_one_sided": alpha_one_sided,
            "target_power": target_power,
            "achieved_power_at_frozen_pair_count": achieved_power,
        },
        "runtime": runtime,
        "files": {
            name: {
                "path": path.as_posix(),
                "sha256": _sha256(resolved[name]),
            }
            for name, path in file_paths.items()
        },
        "protected_key_in_manifest": False,
    }


def validate_frozen_manifest(
    manifest_path: Path,
    *,
    repo_root: Path = Path("."),
) -> dict:
    manifest = _load_json_object(manifest_path, label="design_freeze_manifest")
    expected_manifest_fields = {
        "schema_version",
        "status",
        "git_commit",
        "model_id",
        "model_revision",
        "treatment_version",
        "audit_rate_ppm",
        "confirmatory_seeds",
        "scenario_schema",
        "runtime_schema",
        "number_of_confirmatory_pairs",
        "sample_size_planning",
        "runtime",
        "files",
        "protected_key_in_manifest",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("manifest fields do not match frozen schema")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported H4-C design-freeze manifest schema")
    if manifest.get("status") != "frozen-before-confirmatory-data":
        raise ValueError("manifest is not in frozen-before-confirmatory-data state")
    if manifest.get("protected_key_in_manifest") is not False:
        raise ValueError("manifest must explicitly exclude protected key material")
    if _contains_forbidden_key(manifest):
        raise ValueError("manifest contains a forbidden protected-key field")

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != _REQUIRED_FILE_ROLES:
        raise ValueError("manifest file roles do not match frozen schema")

    for role, entry in files.items():
        if not isinstance(entry, dict):
            raise ValueError(f"manifest file entry is invalid: {role}")
        path_value = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected, str):
            raise ValueError(f"manifest file entry is incomplete: {role}")
        path = Path(path_value)
        expected_path = _CANONICAL_FILE_PATHS.get(role)
        if expected_path is not None and path != expected_path:
            raise ValueError(
                f"manifest {role} path is not the canonical execution path"
            )
        resolved = _resolve_repo_file(repo_root, path, label=role)
        actual = _sha256(resolved)
        if actual != expected:
            raise ValueError(f"frozen file hash mismatch: {role}")
        committed = _sha256_at_commit(Path("."), manifest["git_commit"], path)
        if committed != expected:
            raise ValueError(
                f"declared repository commit does not contain frozen {role} bytes"
            )

    runtime_path = repo_root / Path(files["runtime_config"]["path"])
    runtime = validate_runtime_config(
        _load_json_object(runtime_path, label="runtime_config")
    )
    if runtime != manifest.get("runtime"):
        raise ValueError("runtime config does not match manifest-bound runtime")
    if manifest.get("treatment_version") != runtime["treatment_version"]:
        raise ValueError("manifest treatment_version mismatch")
    if manifest.get("audit_rate_ppm") != runtime["audit_rate_ppm"]:
        raise ValueError("manifest audit_rate_ppm mismatch")
    if manifest.get("confirmatory_seeds") != runtime["seeds"]:
        raise ValueError("manifest seed list mismatch")

    scenario_path = repo_root / Path(files["scenario_set"]["path"])
    scenario_count = _scenario_count(scenario_path)
    expected_pairs = scenario_count * len(runtime["seeds"])
    if manifest.get("number_of_confirmatory_pairs") != expected_pairs:
        raise ValueError("manifest pair count no longer matches frozen design")

    planning = manifest.get("sample_size_planning")
    if not isinstance(planning, dict):
        raise ValueError("manifest sample-size planning is missing")
    if set(planning) != {
        "alternative",
        "p10_c1_success_c2_failure",
        "p01_c1_failure_c2_success",
        "alpha_one_sided",
        "target_power",
        "achieved_power_at_frozen_pair_count",
    }:
        raise ValueError("manifest sample-size planning fields are invalid")
    if planning.get("alternative") != "C1 SAER > C2 SAER":
        raise ValueError("manifest primary alternative mismatch")
    p10 = planning.get("p10_c1_success_c2_failure")
    p01 = planning.get("p01_c1_failure_c2_success")
    alpha = planning.get("alpha_one_sided")
    power = planning.get("target_power")
    achieved = planning.get("achieved_power_at_frozen_pair_count")
    if not isinstance(p10, (int, float)) or isinstance(p10, bool):
        raise ValueError("manifest p10 is invalid")
    if not isinstance(p01, (int, float)) or isinstance(p01, bool):
        raise ValueError("manifest p01 is invalid")
    if not 0.0 <= p01 < p10 <= 1.0 or p10 + p01 > 1.0:
        raise ValueError("manifest planning probabilities are invalid")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0 < alpha < 1:
        raise ValueError("manifest alpha is invalid")
    if not isinstance(power, (int, float)) or isinstance(power, bool) or not 0 < power < 1:
        raise ValueError("manifest target power is invalid")
    if not isinstance(achieved, (int, float)) or isinstance(achieved, bool):
        raise ValueError("manifest achieved power is invalid")
    recomputed_power = exact_directional_mcnemar_power(
        expected_pairs,
        p10=float(p10),
        p01=float(p01),
        alpha=float(alpha),
    )
    if abs(float(achieved) - recomputed_power) > 1e-12:
        raise ValueError("manifest achieved power does not match frozen planning inputs")
    if recomputed_power < float(power):
        raise ValueError("manifest pair count does not satisfy target power")
    manifest_git_commit = str(manifest.get("git_commit", "")).strip().lower()
    resolved_manifest_commit = _resolve_git_commit(manifest_git_commit, Path("."))
    if manifest_git_commit != resolved_manifest_commit:
        raise ValueError("manifest git commit is not canonical")
    if not str(manifest.get("model_id", "")).strip():
        raise ValueError("manifest model_id is invalid")
    if not str(manifest.get("model_revision", "")).strip():
        raise ValueError("manifest model_revision is invalid")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an H4-C design-freeze manifest")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
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
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path("experiments/h4c_model_in_loop.py"),
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("experiments/h4c_local_hf_adapter.py"),
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("experiments/h4c_local_model_confirmatory.py"),
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("experiments/h4c_paired_analysis.py"),
    )
    parser.add_argument(
        "--power-helper",
        type=Path,
        default=Path("experiments/h4c_power.py"),
    )
    parser.add_argument(
        "--generator-runtime",
        type=Path,
        default=Path("dualstream/generator.py"),
    )
    parser.add_argument(
        "--schedule-implementation",
        type=Path,
        default=Path("dualstream/compact_evidence.py"),
    )
    parser.add_argument(
        "--freeze-validator",
        type=Path,
        default=Path("scripts/h4c_freeze_manifest.py"),
    )
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    try:
        manifest = build_manifest(
            git_commit=args.git_commit,
            model_id=args.model,
            model_revision=args.model_revision,
            number_of_pairs=args.number_of_pairs,
            planning_p10=args.planning_p10,
            planning_p01=args.planning_p01,
            alpha_one_sided=args.alpha_one_sided,
            target_power=args.target_power,
            preregistration=args.preregistration,
            scenario_set=args.scenario_set,
            runtime_config=args.runtime_config,
            evaluator=args.evaluator,
            adapter=args.adapter,
            runner=args.runner,
            analysis=args.analysis,
            power_helper=args.power_helper,
            generator_runtime=args.generator_runtime,
            schedule_implementation=args.schedule_implementation,
            freeze_validator=args.freeze_validator,
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
