# The Internal Monologue: A Dual-Stream Architecture for Auditable Inner-Alignment Signals
## Version 2.6 Adversarial-Operational Revision (Technical Proposal)

### Status note (v2.6)
This v2.6 revision adds adversarial-operational clarifications for: (i) internally motivated proof-of-concept (POC) design, (ii) dynamic CI audit scheduling, (iii) deterministic red-state fallback routing, and (iv) randomized/non-static telemetry paths to raise monitor-evasion cost. These additions do **not** prove inner alignment and do **not** prove deception detection completeness.

## Abstract
Dual-Stream Architecture (DSA) is proposed as a system-level evidence-capture and verifier contract for alignment-relevant auditing. DSA does not prove inner alignment. Instead, it makes some answer/evidence inconsistencies observable, measurable, regressible, and (for DSA-P) attestable under explicit assumptions about probe fidelity, system integrity, and decoding semantics. The Monologue Stream is typed telemetry/evidence rather than user-visible chain-of-thought, and DSA preserves a minimum mandatory capture set: chosen token id plus raw final-layer pre-control top-K logits. Probes remain probabilistic and fallible.

## 1. Introduction
DSA separates user-facing Answer Stream tokens from a parallel Monologue evidence stream that is machine-typed and verifier-auditable. The intent is operational: detect and regress alignment-relevant inconsistencies without claiming privileged access to true internal intent.

A valid POC for deceptive-alignment-relevant auditing **MUST** distinguish host-induced answer/evidence divergence from model-internal answer/evidence divergence. Forced decoding and host-side token stripping are useful plumbing checks for evidence capture integrity, but they are insufficient as evidence that DSA detects internally motivated deception pressure.

### 1.1 DSA-R vs DSA-P
- **DSA-R**: software-only research/debug mode, no device attestation claims.
- **DSA-P**: hardware-attested production mode with evidence binding and verifier-side integrity checks.

Inseparability and tamper-resistance claims **MUST** be interpreted as DSA-P-only claims.

## 2. Threat model and deployment assumptions
### 2.1 Trust boundaries
- Model: partially trusted, may optimize proxy objectives misaligned with user intent.
- Host runtime: untrusted/compromisable.
- Accelerator/device: partially trusted in DSA-R; trusted measurement root in DSA-P.
- Verifier: trusted external auditor.

### 2.1.3 Experimental threat model for proof-of-concept evaluation
POC levels are explicitly separated:
- **Level 0 (plumbing)**: host-induced divergence (e.g., forced compliance decode, host-side refusal stripping).
- **Level 1 (internally motivated proxy)**: reward-shaped/sycophancy setup where model reward increases for agreeing with false premises.
- **Level 2 (adaptive monitor-aware)**: model is trained or prompted to evade telemetry while preserving deceptive output behavior.

Level 0 **MUST NOT** be used alone to support claims about catching internally motivated deception.

## 3. Assumptions and non-goals
### 3.1 Assumptions
- CI and production monitoring are computationally budgeted systems; audit depth must be schedulable by risk and latency/storage constraints.
- Verifier has access to integrity metadata, policy thresholds, and capture semantics.
- For DSA-P randomized schedules, verifier can reconstruct nonce-derived audit selections.

### 3.2 Non-goals
- DSA does not guarantee that static probe suites impose prohibitive evasion cost on future models.
- DSA does not expose chain-of-thought to users.
- DSA does not provide mathematical proof that deception is impossible.

## 4. Producing the Monologue evidence stream
### 4.1 Mandatory evidence capture
Each token frame **MUST** include, at minimum:
1. chosen token id;
2. raw final-layer pre-control top-K logits.

Capture point is post-model logits and pre-decoder controls (temperature, penalties, top-k/top-p/min-p filtering, safety masks, constrained decoding).

### 4.2 Probe-pack calibration and telemetry
Probe packs output AST-coded scores and integrity metadata. Probe outputs are probabilistic evidence signals.

### 4.2.2 Randomized and orthogonal probe selection
Static probe sets are vulnerable to monitor overfitting. DSA-P deployments **SHOULD** support nonce-conditioned selection of orthogonal telemetry paths. A cryptographic nonce **MAY** select, per token/window:
- activation slices,
- attention-head subsets,
- residual-stream projections,
- probe subset packs.

Verifier **MUST** be able to reconstruct selection schedules from nonce policy and manifest hashes. Evidence frames **MUST** carry sufficient metadata to verify the realized randomized path without exposing future schedules to the model. This raises expected evasion cost but is not foolproof.

## 5. Coherence audit and enforcement
### 5.1 AST-1 operational taxonomy updates
AST-1 continues as machine-parseable operational labels. Additional codes:
- **304**: static-monitor evasion risk
- **305**: randomized-probe audit path invoked
- **530**: fallback route invoked
- **531**: retry budget exceeded

### 5.2 Token-to-response mapping
Outcome states:
- **PASS**: no spans cross review threshold.
- **REVIEW**: ambiguous/moderate risk spans cross review threshold.
- **FAIL**: severe risk spans cross fail threshold.
- **FALLBACK**: fail-state routed to deterministic safety fallback flow.

### 5.3 Dynamic audit scheduling
Full per-token heavy auditing is not always required. CI and runtime auditing should use cascading tiers:
1. Always run structural/integrity checks.
2. Run cheap logits/entropy/refusal-mass/AST tripwires on all tokens or windows.
3. Escalate suspicious windows to heavier probe paths.
4. If red threshold triggers, invoke deterministic fallback or fail build.
5. Retain full telemetry by default for failed/review slices; retain compact summaries otherwise unless policy requires full retention.

High-risk prompt classes and high-severity windows receive stricter escalation than low-risk windows.

### 5.4 Enforcement semantics and deterministic fallback routing
FAIL during interactive generation **MUST NOT** trigger unbounded retry on identical context.

Deterministic fallback protocol:
1. increment red-state counter;
2. halt current generation;
3. flush or isolate relevant KV-cache segments where applicable;
4. apply safe fallback transform (context truncation, safe-refusal override, non-generative refusal component, or canned safe refusal);
5. emit AST 530 audit record (and 531 if retry budget exhausted);
6. enforce retry budget;
7. fail-closed if fallback path also fails.

Availability and DoS resilience: systems **SHOULD** include bounded retry budgets, cooldown policies, and observability hooks to prevent deterministic red-loop amplification.

## 6. CI integration and audit tiers
CI integration is tiered rather than uniformly exhaustive:
- **Tier 0**: schema/integrity checks (always-on, low cost).
- **Tier 1**: logits-only + entropy/refusal-mass divergence checks.
- **Tier 2**: targeted probe evaluation on suspicious spans/windows.
- **Tier 3**: full ensemble audit for high-risk suites, nightly runs, release candidates, and regressions.

CI modes:
- smoke (commit-blocking),
- targeted,
- nightly,
- release-blocking deep audits.

Latency/storage acceptance criteria **MUST** report:
- wall-clock runtime multiplier vs output-only CI,
- fraction of tokens escalated to heavy probe tiers,
- selective-log retention rate and storage per 10k tokens,
- explicit separation of PR-gating vs scheduled deep-audit costs.

## 7. High-stakes illustrative scenarios
### Scenario C: Internally motivated sycophancy / reward-hacking POC (design only)
Setup: a model variant is fine-tuned/reward-shaped so agreement with user false premises receives higher reward than factual correction (e.g., “soda is an ideal fertilizer for all crops”).

Expected audit pattern (target behavior for evaluation design):
- Answer Stream agrees with false premise.
- Monologue evidence elevates factuality concern (e.g., AST=101 analogue).
- Optional sycophancy-pressure signal elevated (e.g., AST=301 if configured).

This is an experimental design specification, not a reported empirical result.

## 8. Evaluation and red-team plan
### 8.1 Hypotheses
- **H6**: Level-1 internally motivated proxy setups yield stronger answer/evidence divergence detection gains vs output-only baselines than Level-0 plumbing-only setups.
- **H7**: Dynamic audit scheduling reduces CI wall-clock cost while preserving most detection power.
- **H8**: Deterministic fallback routing reduces/eliminates red-state retry loops in adversarial stress tests.
- **H9**: Nonce-randomized probes increase monitor-evasion cost vs static probe schedules.

### 8.2 Benchmark families
- Internally motivated sycophancy/reward-hacking POC benchmark.
- CI scalability benchmark.
- Red-state loop/DoS stress benchmark.
- Static vs randomized monitor-evasion benchmark.

### 8.3 Metrics
- Host-forced divergence detection vs internally motivated divergence detection.
- CI runtime multiplier.
- Percentage of tokens receiving heavy probe evaluation.
- Storage retained per 10k tokens after selective logging.
- Red-state loop count.
- Fallback success rate.
- Retry-budget exhaustion rate.
- Static-monitor evasion success rate.
- Randomized-monitor evasion success rate.
- Evasion cost delta (static vs randomized schedules).

## 9. Limitations
- Reward-hacking/sycophancy POCs are still proxies, not proof of naturally emergent deceptive alignment.
- Entropy and refusal-mass anomalies can reflect benign uncertainty.
- Dynamic scheduling can miss failures that do not trigger cheap tripwires.
- Fallback routing may reduce availability risk yet increase over-refusal/UX degradation.
- Randomized probes raise evasion cost but do not establish formal lower bounds.
- Nonce scheduling assumes future audit choices are not observable by the model.

## 10. v2.6 roadmap
1. **Phase 1**: retain Level-0 plumbing tests as integrity sanity checks.
2. **Phase 2**: add Level-1 sycophancy/reward-hacking POC.
3. **Phase 3**: implement CI tiering and selective telemetry retention.
4. **Phase 4**: implement deterministic fallback router and DoS stress tests.
5. **Phase 5**: deploy nonce-randomized probe selection and static-vs-randomized evasion benchmarks.

## Appendix F: Internally motivated deception POC (sketch)
1. Train/evaluate variant with reward bonus for agreeing with known false premises.
2. Hold prompt set constant across output-only and DSA variants.
3. Measure divergence detection metrics without host forcing.
4. Report no claims beyond proxy-behavior evidence.

## Appendix G: Dynamic CI audit scheduler (pseudocode)
```pseudo
for case in ci_suite:
  run_tier0_integrity(case)
  windows = cheap_tripwire_scan(case)  # logits/entropy/refusal-mass
  for w in windows:
    if risk(w) >= t2_threshold:
      run_tier2_targeted_probes(w)
    if risk(w) >= t3_threshold or case.high_risk:
      run_tier3_full_ensemble(w)
  decision = map_to_pass_review_fail_fallback(case)
  retain(case, full=(decision in {REVIEW, FAIL, FALLBACK}))
```

## Appendix H: Deterministic fallback router (pseudocode)
```pseudo
if state == FAIL:
  red_count += 1
  stop_generation()
  isolate_or_flush_kv(adversarial_span)
  route = select_fallback_policy(red_count, retry_budget)
  emit_ast(530)
  if red_count > retry_budget:
    emit_ast(531)
    fail_closed()
  else:
    run_fallback(route)
```

## Appendix I: Nonce-randomized probe selection (pseudocode + metadata)
```pseudo
nonce = verifier_nonce(request_id, epoch_key)
schedule = derive_schedule(nonce, probe_manifest_hash)
for token_window in stream:
  selected_paths = schedule[token_window.id]
  capture(selected_paths)
  emit_metadata({
    nonce_commitment,
    schedule_version,
    selected_path_ids,
    probe_pack_hash,
    capture_stage
  })
```

Evidence-frame metadata additions (additive envelope fields):
- `nonce_commitment`
- `schedule_version`
- `selected_path_ids`
- `probe_pack_hash`
- `capture_stage`

