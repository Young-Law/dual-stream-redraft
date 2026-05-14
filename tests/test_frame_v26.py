from dualstream.frame import MonologueFrameV1, TopKToken

def test_frame_v26_json_fields():
    fr = MonologueFrameV1(prompt_nonce=1, token_index=0, chosen_id=2, topk=[TopKToken(2, 0.9)], ast_signals=[{"ast_code":101,"score":0.8,"provenance":"heuristic"}], retry_budget_remaining=1)
    d = fr.to_dict()
    assert d['assurance_class'] == 'DSA-R'
    assert d['capture_stage_id'] == 'raw_final_layer_pre_control_logits'
    assert d['ast_signals'][0]['ast_code'] == 101
