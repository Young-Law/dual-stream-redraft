from dualstream import vocab

def test_ast_codes_and_labels_present():
    assert vocab.AST_VOCAB[101] == 'premise likely false'
    assert vocab.AST_VOCAB[531] == 'retry budget exceeded'
    assert all(100 <= c <= 599 for c in vocab.AST_VOCAB)

def test_legacy_mapping():
    assert vocab.to_ast_code(vocab.FACTUALITY_CONCERN) == 101
    assert vocab.to_ast_code(vocab.RETRY_BUDGET_EXCEEDED) == 531
