# JSS Submission Package Notes

## Manuscript positioning

For the Journal of Systems and Software submission, frame the paper as a software-architecture and verification contribution for AI-system audit pipelines. The evidence contribution is mechanism-level rather than a semantic proof of deceptive-alignment detection.

## H4-B result statement

Corrected H4-B holds the public replay context constant across paired schedules (`audit_key_id=7`) and varies only the secret key material. A placement attacker that exploits the predictable public schedule eliminates public-schedule exposure for 1-, 10-, and 50-token events and reduces 100-token exposure to 0.00056, while protected keyed exposure remains near the key-blind analytical expectation (0.00517, 0.04981, 0.22336, 0.39370, and 0.91905 for lengths 1, 10, 50, 100, and 500). The result supports schedule-specific synthetic evasion resistance when the protected audit key is unknown.

## Required claim boundary

Do not state or imply that H4-B establishes:
- semantic deceptive-alignment detection;
- semantic validity of model logits;
- probe correctness;
- robustness to representation-level model adaptation.

## Recommended paper restructuring

1. Move completed H4-A/H4-B evidence out of the Evaluation and Red-Team Plan into a dedicated Results section.
2. Separate implemented controls from roadmap items.
3. Report the H4-B result and scope boundary in the abstract.
4. Keep the 500-token no-placement-freedom boundary explicit in Results and Limitations.
5. Preserve corrected workflow run #9, head `620101bb7c91b8447956d0a77ce9e4d8fb584499`, result JSON SHA-256 `b419fc2c833b407bdbe59f3780bcb89f364244e0903cd2045a6ee7206adbbc61`, and the GitHub Actions artifact digest in a reproducibility subsection.
6. Leave model-in-the-loop adaptive evasion as future work until it has its own preregistered experiment.

## Cover-letter emphasis

Emphasize:
- software architecture;
- verification and validation;
- CI-backed conformance testing;
- portable deterministic verifier work;
- protected keyed sampling;
- retention assurance;
- explicit evidence boundaries and reproducibility.

Avoid presenting the work as proof that deceptive alignment has been solved or directly detected.
