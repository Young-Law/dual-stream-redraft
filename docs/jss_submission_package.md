# JSS Submission Package Notes

## Manuscript positioning

For the Journal of Systems and Software submission, frame the paper as a software-architecture and verification contribution for AI-system audit pipelines. The evidence contribution is mechanism-level rather than a semantic proof of deceptive-alignment detection.

## H4-B result statement

H4-B shows that a placement attacker that can exploit a predictable public sampling schedule nearly eliminates public-schedule exposure for events up to 100 tokens, while the same placement strategy retains exposure near the key-blind analytical expectation under protected keyed sampling. The result supports schedule-specific synthetic evasion resistance when the protected audit key is unknown.

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
5. Preserve the workflow run, commit SHA, and artifact digest in a reproducibility subsection.
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
