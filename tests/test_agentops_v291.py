from dualstream.agentops import *
from dualstream.vocab import *

def reg():
    return ToolRegistryManifest([ToolSpec('pay','Pay','send money','in1','out1','high-impact','non-idempotent','payments:write','required','ops','1')])

def test_agent_trajectory_event_serialization_and_context_snapshot_roundtrip():
    e=context_snapshot(active_instructions_hash='h',token_count=10,max_tokens=20,request_tenant_id='t')
    assert AgentTrajectoryEvent.from_json(e.to_json()).event_type=='ContextSnapshot'

def test_tool_registry_manifest_hashing_and_validation():
    r=reg(); assert len(r.digest())==64; assert r.get('pay').name=='Pay'

def test_unauthorized_tool_scope_high_impact_non_idempotent_retry_failures():
    e=AgentTrajectoryEvent('ToolCall','s','actor',payload={'tool_id':'pay','input_schema_hash':'in1','retry_count':1})
    out=validate_agent_trajectory([e],reg(),allowed_scopes=['read'])
    codes={f['ast_code'] for f in out['findings']}; assert {AST_UNAUTHORIZED_TOOL_SCOPE,AST_HIGH_IMPACT_ACTION_WITHOUT_APPROVAL,AST_NON_IDEMPOTENT_RETRY_RISK} <= codes

def test_unknown_schema_invalid_unsanitized_tool_result():
    events=[AgentTrajectoryEvent('ToolCall','s','a',payload={'tool_id':'missing'}),AgentTrajectoryEvent('ToolCall','s2','a',payload={'tool_id':'pay','input_schema_hash':'bad','approval_id':'ok'}),AgentTrajectoryEvent('ToolResult','s2','a',payload={'sanitized':False,'reentered_model_context':True})]
    codes={f['ast_code'] for f in validate_agent_trajectory(events,reg(),allowed_scopes=['payments:write'])['findings']}; assert {AST_UNKNOWN_TOOL,AST_SCHEMA_INVALID_TOOL_CALL,AST_TOOL_OUTPUT_INJECTION_RISK} <= codes

def test_cross_tenant_memory_and_write_without_provenance_failure():
    events=[AgentTrajectoryEvent('MemoryRead','m','a',payload={'tenant_id':'a','request_tenant_id':'b'}),AgentTrajectoryEvent('MemoryWrite','mw','a',payload={'tenant_id':'a','request_tenant_id':'a'})]
    codes={f['ast_code'] for f in validate_agent_trajectory(events,reg())['findings']}; assert {AST_CROSS_TENANT_MEMORY_RISK,AST_MEMORY_WRITE_WITHOUT_PROVENANCE} <= codes

def test_rollout_gate_and_incident_to_evaluation_check():
    events=[AgentTrajectoryEvent('Rollout','r','ops',payload={'gate_passed':False}),AgentTrajectoryEvent('Incident','i','ops',payload={'incident_id':'inc1'})]
    codes={f['ast_code'] for f in validate_agent_trajectory(events,reg())['findings']}; assert {AST_ROLLOUT_GATE_FAILURE,AST_INCIDENT_NOT_CONVERTED_TO_EVALUATION_CASE} <= codes

def test_all_event_types_declared():
    assert EVENT_TYPES == ['ContextSnapshot','ToolIntent','ToolCall','ToolResult','MemoryRead','MemoryWrite','Delegation','Stop','Escalation','Rollout','Incident','EvaluationResult']
