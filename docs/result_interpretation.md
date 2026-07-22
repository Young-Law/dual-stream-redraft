# Result interpretation

Implementation validity asks whether the DSA software records, serializes, reconstructs, verifies, budgets, and rejects evidence correctly. Signal validity asks whether the evidence is scientifically useful for detecting or auditing target behaviors. Passing implementation-validity checks is necessary engineering evidence, but it is not proof of signal validity.

Authentic telemetry means the recorded artifact is bound to the implemented generation or replay process and passes integrity checks. Semantically faithful telemetry is stronger: it requires evidence that the telemetry meaningfully represents the model behavior under study. The current synthetic conformance run addresses authenticity and reconstruction, not semantic faithfulness.

Synthetic conformance tests use controlled token streams to isolate evidence-format overhead and verifier behavior. Behavioral benchmarks require real prompts, model outputs, labels, calibration splits, and untouched test splits.

Exploratory experiments may help discover failure modes or candidate thresholds. Confirmatory experiments must use a frozen protocol and must not tune thresholds on test data.

Calibration data may set thresholds such as a fixed benign false-positive target. Test data must remain untouched until final evaluation.

Negative results and failed runs must be retained because they define the actual reliability envelope of the system and prevent selective reporting.

Passing integrity checks does not prove alignment. It only shows that an artifact is structurally consistent with the verifier rules that were applied.
