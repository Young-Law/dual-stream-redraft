import json, subprocess, sys
from dualstream.compact_evidence import encode_compact_sequence


def test_cli_verify_evidence_budget_json(tmp_path):
    (tmp_path / "compact_evidence.dsae").write_bytes(encode_compact_sequence([{"chosen_id": i, "topk_ids": [i,i+1,i+2], "topk_scores": [.7,.2,.1]} for i in range(500)]))
    proc = subprocess.run([sys.executable, "-m", "dualstream.cli", "verify-evidence-budget", "--artifact", str(tmp_path), "--profile", "DSA-CI-Lite", "--ci-mode", "pr", "--json"], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert json.loads(proc.stdout)["ok"] is True


def test_documented_budget_verification_command_passes(tmp_path):
    import subprocess, sys
    run = tmp_path / "doc_budget"
    run.mkdir()
    (run / "compact_evidence.dsae").write_bytes(encode_compact_sequence([{"chosen_id": i, "topk_ids": [i,i+1,i+2], "topk_scores": [.7,.2,.1]} for i in range(10000)]))
    proc = subprocess.run([sys.executable, "-m", "dualstream.cli", "verify-evidence-budget", "--artifact", str(run), "--profile", "DSA-CI-Lite", "--ci-mode", "pr", "--json"], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert json.loads(proc.stdout)["ok"] is True


def test_cli_strict_budget_failure_json_status(tmp_path):
    rows = [{"chosen_id": i + 100000, "topk_ids": [i,i+1,i+2], "topk_scores": [.7,.2,.1]} for i in range(4)]
    (tmp_path / "compact_evidence.dsae").write_bytes(encode_compact_sequence(rows, adaptive_k=False))
    proc = subprocess.run([sys.executable, "-m", "dualstream.cli", "verify-evidence-budget", "--artifact", str(tmp_path), "--profile", "DSA-CI-Lite", "--ci-mode", "pr", "--strict-profile-budget", "--json"], text=True, capture_output=True)
    assert proc.returncode != 0
    payload = json.loads(proc.stdout)
    assert payload["budget_status"] == "fail"


def test_cli_short_fixture_non_strict_json_status(tmp_path):
    rows = [{"chosen_id": i + 100000, "topk_ids": [i,i+1,i+2], "topk_scores": [.7,.2,.1]} for i in range(4)]
    (tmp_path / "compact_evidence.dsae").write_bytes(encode_compact_sequence(rows, adaptive_k=False))
    proc = subprocess.run([sys.executable, "-m", "dualstream.cli", "verify-evidence-budget", "--artifact", str(tmp_path), "--profile", "DSA-CI-Lite", "--ci-mode", "pr", "--json"], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert json.loads(proc.stdout)["budget_status"] == "not_evaluated_short_fixture"
