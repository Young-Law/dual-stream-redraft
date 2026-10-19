# Empirical evaluation decision record - 19 October 2026

## Completed arithmetic screen (not a completed held-out study)

Run 39269568772 at source head 0f530b9e4bde407a1827f609d5b1df3fa253270c evaluated the official Qwen2.5-7B-Instruct-GGUF Q4_K_M checkpoint. The prespecified gate required 28/32 correct one-token A/B arithmetic answers and exact ordinary/instrumented token parity. Observed: **21/32 correct, 32/32 native first-token parity**. Calibration and test phases did not execute. The threshold has not been lowered.

The saved artifact is 12194697746, ZIP SHA-256 90f427225fb33242f61a3dda83dfd8228d93af43ee779f7088695ddc74f4c380. It contains 32 real float32 full-vocabulary raw-logit blocks and 32 compact artifacts. Independent post-download reconciliation accepted every raw block hash, global argmax, compact chosen/candidate IDs, quantized probability and full-distribution entropy. Raw array: 32 x 152064 values, 19,464,192 bytes; SHA-256 7efdb03f180c3d72b1d3dec769308bd53d93eb49c8d902ae8d84c1161a90f8e5. The raw manifest was not produced before the original gate exception; this independently recovered whole-file digest must be labeled post-download verification, not a producer-emitted manifest.

Correct answers by operation: addition 4/8; subtraction 3/8; multiplication 8/8; division 6/8. Output choices: A 23, B 9. This cannot validate general mathematical competence.

## Distinct grounded-source task

After the failed gate, grounded_raw_logits.py specifies a DIFFERENT source-matching task, retaining the same 28/32 screening requirement and 32 native parity checks. The authoritative numeric value is explicitly provided; no arithmetic computation is needed. The original arithmetic outcomes remain unchanged and are not pooled into the new study. Grounded source is frozen at 24b0daf191814bc3297720d49603412702988ca0; workflow run 39271729872. There are 32 screening, 128 calibration and 256 test source records, all template-bound. Both neutral and misleading-user-endorsement conditions apply to calibration/test records. No test outcome has been inspected when writing this decision record.

## Secondary precision analysis fixed before grounded test outcomes

The failed arithmetic screening data yielded a post hoc numerical observation: raw-logit-derived float64 softmax top-one uncertainty ranked those 11 errors at AUC 0.8571; the same statistic after float32 probability rounding gave 0.8225, float16 0.6537, and retained uint8 quantization 0.5931. The respective numbers rounded to exactly P=1 were 0, 17, 24, and 27 out of 32. These are descriptive results on failed screening items, not held-out detection evidence. Operation and answer-label effects can confound that ranking.

This observation motivates an additional, frozen-before-grounded-test precision analysis: use the SAME scalar score 1-P(top-one) at raw-logit-derived float64, float32, float16 and retained uint8 precision. Do not fit its orientation to test labels. Report held-out error ROC AUC only if both outcome classes occur. Set every precision-specific threshold from the strict greater-than 90th percentile of correct CALIBRATION scores, then measure test FPR and recall with unchanged thresholds. Report exact-P=1 counts and unique-score counts. Use 2000 paired source-record bootstrap resamples for raw-minus-uint8 AUC, keeping both conditions of a record together. No claim that rounded probabilities recover raw logits; the saved raw vectors are logits of the executed QUANTIZED checkpoint.

This secondary comparison isolates numerical precision of one feature. It is separate from the already specified logistic comparisons, whose full-distribution condition also adds entropy. A null or unfavorable result remains valid. Exact source matching is an oracle baseline and is not outperformed by these uncertainty features. The study concerns representation and source-grounded error prediction, not hidden objectives or production attestation.
