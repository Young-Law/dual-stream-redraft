from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import hashlib, json
from .vocab import *
from .context_serialization import verify_context_serialization

def canon(o:Any)->bytes: return json.dumps(o,sort_keys=True,separators=(',',':')).encode()
def sha(o:Any)->str: return hashlib.sha256(canon(o)).hexdigest()

@dataclass
class TokenSpan: start:int; end:int
@dataclass
class AgentTrajectoryEvent:
    event_type:str; agent_step_id:str; actor_id:str; token_span:Optional[TokenSpan]=None; payload:Dict[str,Any]=field(default_factory=dict)
    def to_dict(self):
        d=asdict(self); d['token_span']=asdict(self.token_span) if self.token_span else None; return d
    @classmethod
    def from_dict(cls,d):
        ts=d.get('token_span'); return cls(d['event_type'],d['agent_step_id'],d['actor_id'],TokenSpan(**ts) if ts else None,dict(d.get('payload') or {}))
    def to_json(self): return canon(self.to_dict()).decode()
    @classmethod
    def from_json(cls,s): return cls.from_dict(json.loads(s))
EVENT_TYPES=['ContextSnapshot','ToolIntent','ToolCall','ToolResult','MemoryRead','MemoryWrite','Delegation','Stop','Escalation','Rollout','Incident','EvaluationResult']

def context_snapshot(agent_step_id='s1',actor_id='agent',**payload): return AgentTrajectoryEvent('ContextSnapshot',agent_step_id,actor_id,payload=payload)

@dataclass
class ToolSpec:
    tool_id:str; name:str; description:str; input_schema_hash:str; output_schema_hash:str; side_effect_class:str; idempotency_class:str; permission_scope:str; approval_requirement:str; owner:str; version:str; deprecation_state:str='active'; policy_binding:Optional[str]=None
@dataclass
class ToolRegistryManifest:
    tools:List[ToolSpec]; version:str='tool-registry-v2.9.1'
    def to_dict(self): return {'version':self.version,'tools':[asdict(t) for t in self.tools]}
    def digest(self): return sha(self.to_dict())
    def get(self,tid): return next((t for t in self.tools if t.tool_id==tid),None)

@dataclass
class AgentOpsControlPlaneManifest:
    source_version:str; model_endpoint_or_build:str; prompt_version:str; tool_registry_version:str; tool_schema_versions:Dict[str,str]; memory_schema_version:str; context_policy_version:str; guardrail_ast_schema_version:str; evaluation_dataset_version:str; verifier_version:str; evidence_profile:str; budget_profile:str; rollout_configuration:Dict[str,Any]; feature_flags:Dict[str,bool]
    def to_dict(self): return asdict(self)
    def digest(self): return sha(self.to_dict())

def validate_agent_trajectory(events:List[AgentTrajectoryEvent], registry:ToolRegistryManifest|None=None, manifest:AgentOpsControlPlaneManifest|None=None, allowed_scopes:List[str]|None=None)->Dict[str,Any]:
    findings=[]; allowed=set(allowed_scopes or [])
    incidents=[]; evals=[]; rollout_seen=False
    tool_calls={}
    for e in events:
        p=e.payload
        if e.event_type=='ToolCall':
            tid=p.get('tool_id'); spec=registry.get(tid) if registry else None
            if not spec: findings.append({'ast_code':AST_UNKNOWN_TOOL,'message':'unknown tool'})
            else:
                if p.get('input_schema_hash') and p.get('input_schema_hash')!=spec.input_schema_hash: findings.append({'ast_code':AST_SCHEMA_INVALID_TOOL_CALL,'message':'schema-invalid tool call'})
                if allowed and spec.permission_scope not in allowed: findings.append({'ast_code':AST_UNAUTHORIZED_TOOL_SCOPE,'message':'unauthorized tool scope'})
                if spec.approval_requirement in ('required','high-impact') and not p.get('approval_id'): findings.append({'ast_code':AST_HIGH_IMPACT_ACTION_WITHOUT_APPROVAL,'message':'high-impact action without approval'})
                if spec.idempotency_class=='non-idempotent' and p.get('retry_count',0)>0 and not p.get('retry_policy_id'): findings.append({'ast_code':AST_NON_IDEMPOTENT_RETRY_RISK,'message':'non-idempotent retry without policy'})
            tool_calls[e.agent_step_id]=tid
        if e.event_type=='ToolResult' and p.get('sanitized') is False and p.get('reentered_model_context') is True: findings.append({'ast_code':AST_TOOL_OUTPUT_INJECTION_RISK,'message':'unsanitized tool output re-entered context'})
        if e.event_type=='ContextSnapshot':
            if p.get('is_stale'): findings.append({'ast_code':AST_STALE_CONTEXT_RISK,'message':'stale context'})
            if p.get('trust_mismatch'): findings.append({'ast_code':AST_RETRIEVAL_SOURCE_TRUST_MISMATCH,'message':'context trust mismatch'})
            if p.get('token_count',0)>p.get('max_tokens',10**18): findings.append({'ast_code':AST_CONTEXT_BUNDLE_EXCEEDS_POLICY,'message':'context bundle exceeds policy'})
            if p.get('serialized_context_payloads'):
                findings.extend(verify_context_serialization(p['serialized_context_payloads']).get('findings', []))
        if e.event_type in ('MemoryRead','MemoryWrite'):
            if p.get('tenant_id')!=p.get('request_tenant_id'): findings.append({'ast_code':AST_CROSS_TENANT_MEMORY_RISK,'message':'cross-tenant memory access'})
            if e.event_type=='MemoryWrite' and not p.get('provenance'): findings.append({'ast_code':AST_MEMORY_WRITE_WITHOUT_PROVENANCE,'message':'memory write without provenance'})
        if e.event_type=='Rollout':
            rollout_seen=True
            if not p.get('gate_passed',True): findings.append({'ast_code':AST_ROLLOUT_GATE_FAILURE,'message':'rollout gate failure'})
            if not p.get('rollback_available',True): findings.append({'ast_code':AST_ROLLBACK_UNAVAILABLE,'message':'rollback unavailable'})
        if e.event_type=='Incident': incidents.append(p.get('incident_id'))
        if e.event_type=='EvaluationResult': evals.append(p.get('incident_id'))
    for iid in filter(None,incidents):
        if iid not in evals: findings.append({'ast_code':AST_INCIDENT_NOT_CONVERTED_TO_EVALUATION_CASE,'message':'incident not converted to evaluation case'})
    if manifest and manifest.rollout_configuration.get('required') and not rollout_seen: findings.append({'ast_code':AST_ROLLOUT_GATE_FAILURE,'message':'missing rollout evidence'})
    return {'trajectory_audit_passed':not findings,'findings':findings}
