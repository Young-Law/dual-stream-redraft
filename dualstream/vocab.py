# Default concept vocabulary (illustrative).
# Compatibility note:
# - Existing textual labels are preserved.
# - AST-1 family IDs are represented by stable numeric concept_id ranges.

DEFAULT_CONCEPT_VOCAB = {
    1001: "USER_INTENT:CONFIRMATION_REQUEST",
    2001: "FACTUALITY_CONCERN:PREMISE_LIKELY_FALSE",
    3001: "POLICY_TENSION:HONESTY_vs_PLEASING_USER",
    3101: "DECEPTION_RISK:CODE_SABOTAGE_BACKDOOR",
    3201: "DECEPTION_RISK:CREDENTIAL_HARVESTING_SOCIAL_ENGINEERING",
}

def concept_label(concept_id: int) -> str:
    return DEFAULT_CONCEPT_VOCAB.get(concept_id, f"CONCEPT_{concept_id}")
