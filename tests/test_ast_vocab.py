from dualstream import vocab

def test_ast_codes_and_labels_present():
    assert vocab.AST_VOCAB[101] == 'premise likely false'
    assert vocab.AST_VOCAB[531] == 'retry budget exceeded'
    assert all(100 <= c <= 599 for c in vocab.AST_VOCAB)

def test_legacy_mapping():
    assert vocab.to_ast_code(vocab.FACTUALITY_CONCERN) == 101
    assert vocab.to_ast_code(vocab.RETRY_BUDGET_EXCEEDED) == 531



def test_coherence_affirmation_avoids_substrings_and_negated_answers():
    from dualstream.audit import coherence_outcome
    from dualstream.frame import ConceptScore, MonologueFrameV1, TopKToken
    frame = MonologueFrameV1(
        prompt_nonce=1,
        token_index=0,
        chosen_id=7,
        topk=[TopKToken(7, 0.8)],
        concepts=[ConceptScore(vocab.FACTUALITY_CONCERN, 0.9)],
        audit_tier="tier0",
    )
    for answer in (
        "Yesterday, the claim was incorrect.",
        "No, that is not correct.",
        "The copyright section describes pressure.",
        "I'm not sure the premise is true.",
    ):
        report = coherence_outcome(answer, [frame])
        assert not any(
            finding.kind == "affirmation_vs_factuality_concern"
            for finding in report.findings
        ), answer
    affirmative = coherence_outcome("Yes, that is correct.", [frame])
    assert affirmative.outcome == "FAIL"
    assert any(
        finding.kind == "affirmation_vs_factuality_concern"
        for finding in affirmative.findings
    )
