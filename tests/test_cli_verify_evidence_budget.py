import json, subprocess, sys
from dualstream.compact_evidence import encode_compact_sequence


def test_cli_verify_evidence_budget_json(tmp_path):
    (tmp_path / "compact_evidence.dsae").write_bytes(encode_compact_sequence([{"chosen_id": i, "topk_ids": [i,i+1,i+2], "topk_scores": [.7,.2,.1]} for i in range(500)]))
    proc = subprocess.run([sys.executable, "-m", "dualstream.cli", "verify-evidence-budget", "--artifact", str(tmp_path), "--profile", "DSA-CI-Lite", "--ci-mode", "pr", "--json"], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert json.loads(proc.stdout)["ok"] is True
