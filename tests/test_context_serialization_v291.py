import json
from dualstream.context_serialization import *
from dualstream.agentops import AgentTrajectoryEvent, validate_agent_trajectory, ToolRegistryManifest
from dualstream.compact import CompactSequence, CompactChunk, make_record, encode_sequence, decode_sequence
from dualstream.vocab import AST_MODEL_FACING_JSON_DEPTH_VIOLATION, AST_NESTED_CONTEXT_SERIALIZATION_RISK


def nested_bundle():
    return {
        'agentops': {'run': {'source': {'version': 'abc123'}}},
        'tool_registry': {'tools': [{'id': 'search', 'schemas': {'input': {'hash': 'sha256:in'}}}]},
        'context': {'request': {'summary': 'answer user'}, 'retrieval': {'artifact': {'id': 'doc1', 'hash': 'sha256:doc'}}},
    }


def test_flat_yaml_serializer_output_for_deep_model_context_bundle():
    text = serialize_flat_yaml_context_bundle(nested_bundle(), namespace='agent_context')
    assert 'agent_context.agentops.run.source.version: abc123' in text
    assert 'agent_context.context.retrieval.artifact.hash: "sha256:doc"' in text
    assert '{' not in text and '[' not in text


def test_linter_pass_for_flattened_yaml_representation():
    text = serialize_flat_yaml_context_bundle(nested_bundle())
    report = lint_context_payloads([{'visibility': 'prompt', 'format': 'yaml', 'content': text}])
    assert report.passed
    assert report.flat_yaml_context_bundle_count == 1
    assert report.maximum_yaml_depth <= DEFAULT_MAX_CONTEXT_DEPTH


def test_linter_failure_for_prompt_visible_nested_json_deeper_than_three_levels():
    report = lint_context_payloads([{'visibility': 'prompt', 'format': 'json', 'content': nested_bundle()}])
    assert not report.passed
    assert report.prompt_visible_json_depth_violations == 1
    assert any(f['ast_code'] == AST_MODEL_FACING_JSON_DEPTH_VIOLATION for f in report.findings)


def test_linter_failure_for_evaluator_visible_nested_json_deeper_than_three_levels():
    report = lint_context_payloads([{'visibility': 'evaluator', 'format': 'json', 'content': nested_bundle()}])
    assert not report.passed
    assert report.evaluator_visible_json_depth_violations == 1


def test_linter_failure_for_inline_deep_json_schema_when_reference_exists():
    schema = {'type': 'object', 'properties': {'arg': {'type': 'object', 'properties': {'q': {'type': 'string'}}}}}
    report = lint_context_payloads([{'visibility': 'prompt', 'format': 'json', 'content': {'tool': {'schema': schema}}, 'schema_hash': 'sha256:schema'}])
    assert not report.passed
    assert report.deep_schema_reference_count == 1
    assert any(f['ast_code'] == AST_NESTED_CONTEXT_SERIALIZATION_RISK for f in report.findings)


def test_verifier_ast_emission_for_context_serialization_codes():
    event = AgentTrajectoryEvent('ContextSnapshot','s','actor',payload={'serialized_context_payloads':[{'visibility':'prompt','format':'json','content':nested_bundle()}]})
    out = validate_agent_trajectory([event], ToolRegistryManifest([]))
    assert any(f['ast_code'] == AST_MODEL_FACING_JSON_DEPTH_VIOLATION for f in out['findings'])


def test_json_remains_valid_for_machine_schemas_manifests_and_compact_transport():
    schema = {'type': 'object', 'properties': {'arg': {'type': 'object', 'properties': {'q': {'type': 'string'}}}}}
    machine = lint_context_payloads([{'visibility':'machine','format':'json','content':schema,'schema_hash':'sha256:schema'}])
    assert machine.passed
    seq = CompactSequence('seq', 1, chunks=[CompactChunk(0, [make_record([1,2,3],[.7,.2,.1], 1)], 0)])
    assert decode_sequence(encode_sequence(seq)).sequence_id == 'seq'
    assert json.loads(json.dumps(schema))['type'] == 'object'


def test_repeated_agentops_metadata_nested_context_failure():
    report = lint_context_payloads([{'visibility':'review','format':'json','content':{'packet': {'sequence_id':'s'}}, 'agentops_bound_fields':['sequence_id']}])
    assert any(f['ast_code'] == AST_NESTED_CONTEXT_SERIALIZATION_RISK for f in report.findings)
