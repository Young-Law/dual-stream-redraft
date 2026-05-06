# CHANGELOG v2.6 pipeline

## Summary of v2.6 requirements
- Added tiered scheduler, lightweight risk, fallback routing, randomized audit metadata, and Level-1 sycophancy proxy labeling.

## Files changed
- dualstream/{generator.py,audit.py,audit_scheduler.py,fallback.py,frame.py,vocab.py,cli.py}
- docs/{dsa_operational_clarifications.md,v2.6_pipeline_alignment.md}
- examples/prompts/v26_*.txt
- README.md

## Tests added/updated
- Existing suite exercised; further targeted v2.6 tests can be expanded.

## Known limitations
- Reference implementation and proxy-only claims; no production guarantees.

## Commands run
- python -m pytest
- python -m dualstream.cli generate ...
