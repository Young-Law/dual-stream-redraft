| critique issue | severity | affected section(s) | edit made | remaining limitation |
|---|---|---|---|---|
| POC relied on host-induced forced compliance | High | 1, 2.1.3, 7, Appendix F | Added Level 0/1/2 threat levels; added internally motivated sycophancy/reward-hacking POC design and scenario | Level-1 remains a proxy, not proof of naturally emergent deception |
| CI cost from exhaustive per-token ensemble audits | High | 5.3, 6, Appendix G | Added dynamic tiered scheduling, trigger-based escalation, window-level heavy probing, selective retention, and CI mode separation | Tripwire misses are possible when cheap signals do not fire |
| Red-state retries can loop and cause DoS | High | 5.2, 5.4, 8.3, Appendix H | Added PASS/REVIEW/FAIL/FALLBACK semantics and deterministic fallback protocol with counters and retry budgets | Fallback can increase over-refusal and degrade UX |
| Static correlated probes can be evaded | High | 4.2.2, 5.1, 8.1-8.3, Appendix I | Added nonce-randomized orthogonal probe selection, AST codes 304/305, and static-vs-randomized evasion metrics | Raises evasion cost without guaranteeing impossibility |
| Overstated claims risk | Medium | Status note, Abstract, 3.2, 9 | Repeated constraints that DSA does not prove alignment or complete deception detection | Proposal still depends on assumptions about probe fidelity and integrity |
