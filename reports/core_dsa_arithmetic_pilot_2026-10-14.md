# DSA-R real-model arithmetic pilot — October 14, 2026

**Research status:** exploratory single-model engineering validation. Not preregistered, powered, held-out or confirmatory. Neither deceptive-alignment detection nor DSA-P attestation is established.

## Provenance

- Real-model GitHub Actions run: https://github.com/Young-Law/dual-stream-redraft/actions/runs/38622850702
- Raw run folders, JSONL frames, compact V3.3 evidence, independent verification, fixture, task answers and full pilot-trials.json: https://github.com/Young-Law/dual-stream-redraft/actions/runs/38622850702/artifacts/11921764704
- Fixture SHA-256: 21e5edc5d0abf8f819c8564e2846f7e988f2d65342b30cbdd21a4d552a5267fb
- Fixed protocol and exact scoring code: experiments/core_signal_pilot.py
- Model: Qwen/Qwen2.5-0.5B-Instruct at immutable revision a338b55dd21219a5f4da42bc11a9313d1a27d4cc, CPU, PyTorch 2.5.1, Transformers 4.49.0, greedy seed 11, maximum 10 new tokens.
- DSA instrumentation: native per-token generator, pre-control unrenormalized top-K probabilities (K=3) and first-token full-distribution entropy, compact V3.3 under DSA-CI-Lite; no heuristic concepts, POC injection or calibrated hidden-state probe pack.
- Design: twelve independently solved arithmetic expressions, each as a true/false equality, each in neutral and agreement-pressure framing: 12 x 2 x 2 = 48 tasks. Task order deterministically shuffled. Truth comes from literal Python arithmetic, not a model. Strict YES/NO answer contract; an invalid answer was predefined as incorrect.

## Primary descriptive outcomes

| Independent measurement | Observed |
| --- | ---: |
| Real-model trials completed | 48/48 |
| Locally independently verified DSA-R evidence streams | 48/48 LOCAL_PASS |
| Strict valid YES/NO outputs | 48/48 |
| Correct task outputs | 24/48 (50%) |
| Actual YES outputs | **48/48** |
| False claims incorrectly affirmed (neutral) | 12/12 |
| False claims incorrectly affirmed (pressure) | 12/12 |
| Paired pressure-minus-neutral false-claim affirmative rate | **0 percentage points** |
| True claims incorrectly denied | 0/24 |
| First-token entropy/error ROC AUC | 0.5226 |
| First-token low raw top-1 probability/error ROC AUC | 0.5191 |

The prespecified exploratory 1,000-resample percentile interval (source-problem cluster bootstrap) for the entropy AUC was [0.474, 0.568], including chance. Each emitted YES response contained two generated token IDs (YES and terminal EOS). No first-token confidence/entropy metric showed credible truth discrimination. The model always affirming *both* true and false claims means this model-task combination is insufficient as a subtle alignment-detection benchmark.

## Important post hoc DSA observability finding (not a detection result)

Inspection of all actual trial-level evidence showed a systematic shift in the raw probability of the emitted YES token (ID 14004), despite **identical visible YES answers**. The YES token remained the raw pre-control top-1 candidate and the committed token in every run.

| Ground-truth claim | Matched pairs | Neutral mean raw P(YES) | Pressure mean raw P(YES) | Mean change | Neutral first-token entropy | Pressure first-token entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| True | 12 | 0.82714 | 0.97048 | +0.14334 | 0.47568 | 0.14970 |
| False | 12 | 0.81530 | 0.96983 | +0.15453 | 0.50072 | 0.15281 |
| Combined | 24 | 0.82122 | 0.97016 | **+0.14894** | 0.48820 | 0.15126 |

The probability rose and full-vocabulary first-token entropy fell in **all 24/24 paired comparisons**. Additional *post hoc* 10,000-resample percentile bootstrap intervals, clustering at the twelve original arithmetic expressions (seed 11), were:
- Combined paired P(YES) change: +0.14894, exploratory 95% interval [+0.1206, +0.1784].
- False-claim pairs only: +0.15453, exploratory 95% interval [+0.1277, +0.1825].
- True-claim pairs only: +0.14334, exploratory 95% interval [+0.1109, +0.1751].

The raw DSA evidence therefore revealed changes in *token preference strength* that were absent from output-only YES/NO strings. This is evidence **of observability of a prompt-conditioned probability shift**, not evidence of improved detection of arithmetic errors, hidden motivations, consciousness, strategic deception or mechanistic sycophancy. Pressure wording and context length differ; lexical priming and generic prompt conditioning cannot be ruled out. This post hoc secondary comparison was not in the pre-run primary protocol and must not be misrepresented as confirmatory.

### Reproduction of the post hoc shift

Extract the archived core-signal-pilot/pilot-trials.json file. Each trial row has source problem ID, ground truth, framing, first-token raw top-1 probability, and first-token entropy. Form matched pairs keyed by (problem_id, claim_truth) and subtract neutral from pressure. Cluster bootstrap samples the 12 problem IDs *with replacement* while carrying the true and false pairs together, using random.Random(11) and 10,000 draws. The analysis does not inspect hidden chain-of-thought or rely on model-reported confidence.

For every inference record, DSA's chosen_id agrees between monologue.jsonl and independently decoded V3.3 compact evidence, frame indices are contiguous, and the verifier returned LOCAL_PASS. This is a **local software-integrity result**, not an independently attested host.

## Publication-ready results paragraph (qualified)

> In an exploratory, single-model evaluation, DSA-R captured and independently verified 48 out of 48 token-level evidence streams from a pinned Qwen2.5-0.5B-Instruct model on paired arithmetic true/false questions under neutral and agreement-pressure prompts. The model returned YES on every trial, yielding 50% oracle accuracy and 100% incorrect affirmation of false statements, with no difference in binary affirmative behavior between conditions. Pre-control first-token entropy showed near-chance discrimination of task errors (ROC AUC 0.523; exploratory problem-clustered 95% interval 0.474–0.568). A post hoc paired analysis nonetheless found that the model's raw probability for the first YES token increased from a mean of 0.821 under neutral phrasing to 0.970 under agreement pressure (mean paired increase 0.149; n=24 claim pairs), while the visible answer remained identical. These findings demonstrate sensitivity of independently captured DSA telemetry to prompt-conditioned token-probability changes not apparent from the output string, but do not establish meaningful detection of errors or deceptive alignment.

## Interpretation and next scientific gate

This pilot is **not** a positive test of whitepaper H1 (improved labeled-error detectability) or H7 (probe incremental value). It supports a narrower engineering measurement claim: native DSA exposes and verifies pre-control probability shifts under a known prompt contrast, even when the surface string is invariant.

To claim alignment-relevant detection, first establish model competence on task instances with nondegenerate correctness outcomes, use a disjoint evaluation set and independent labels, compare calibrated evidence-based decision rules against output-only baselines at matched false-positive budgets, assess probe incremental benefit, and report uncertainty and negative findings. Do not pool with the separately frozen H4-C population. One 0.5B model and twelve source expressions are not grounds for generalization.

**Data longevity:** GitHub Actions retains the linked complete experiment ZIP for 14 days by current workflow settings. The pinned protocol, aggregate results, identifiers, and SHA above remain in the repository; external archival of the complete raw ZIP is recommended before expiration if the manuscript cites trial-level figures.
