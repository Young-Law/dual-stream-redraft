from dualstream import vocab

def test_ast_codes_and_labels_present():
    assert vocab.AST_VOCAB[101] == 'premise likely false'
    assert vocab.AST_VOCAB[531] == 'retry budget exceeded'
    assert all(100 <= c <= 899 for c in vocab.AST_VOCAB)
    assert vocab.AST_VOCAB[523] == 'metadata repetition violation'
    assert vocab.AST_VOCAB[600] == 'unknown tool'
    assert vocab.AST_VOCAB[701] == 'cross-tenant memory risk'
    assert vocab.AST_VOCAB[706] == 'nested context serialization risk'
    assert vocab.AST_VOCAB[707] == 'model-facing JSON depth violation'
    assert vocab.AST_VOCAB[803] == 'incident not converted to evaluation case'

def test_legacy_mapping():
    assert vocab.to_ast_code(vocab.FACTUALITY_CONCERN) == 101
    assert vocab.to_ast_code(vocab.RETRY_BUDGET_EXCEEDED) == 531
