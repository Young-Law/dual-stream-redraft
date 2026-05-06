# CHANGELOG v2.6 pipeline

| requirement | implementation files | tests | status | limitation |
|---|---|---|---|---|
| tiered audit scheduling | `dualstream/audit_scheduler.py`, `dualstream/generator.py` | `tests/test_v26_pipeline.py` | done | heuristic risk model |
| fallback routing | `dualstream/fallback.py`, `dualstream/generator.py` | `tests/test_v26_pipeline.py` | done | deterministic canned behavior |
| randomized audit path | `dualstream/randomized_audit.py`, `dualstream/frame.py` | `tests/test_v26_pipeline.py` | done | not a proof of non-evasion |
| coherence expansion | `dualstream/audit.py` | `tests/test_v26_pipeline.py` | done | rule-based signals |
