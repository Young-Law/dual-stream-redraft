from dualstream.compact_evidence import choose_effective_topk, encode_compact_sequence, reconstruct_token_evidence


def test_adaptive_k_widens_when_chosen_outside_base_top3():
    assert choose_effective_topk(4, [1, 2, 3, 4, 5], 3, 5, True) == 4
    blob = encode_compact_sequence([{"chosen_id": 4, "topk_ids": [1,2,3,4,5], "topk_scores": [.4,.3,.2,.05,.05]}], adaptive_k=True, max_adaptive_k=5)
    assert reconstruct_token_evidence(blob)[0]["effective_topk"] == 4


def test_adaptive_k_does_not_widen_when_chosen_inside_base_top3():
    assert choose_effective_topk(2, [1, 2, 3, 4, 5], 3, 5, True) == 3
