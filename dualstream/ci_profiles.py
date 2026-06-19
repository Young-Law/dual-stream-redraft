CI_PROFILES={
 'pr': {'profile':'DSA-CI-Lite','compact_metadata_amortized':True,'raw_bytes_per_token_hard_fail':24,'raw_bytes_per_token_warning':22},
 'nightly': {'profile':'DSA-CI-Standard','compact_metadata_amortized':True},
 'release-blocking': {'profile':'DSA-CI-Standard','compact_metadata_amortized':True},
 'forensic': {'profile':'DSA-Forensic','raw_bytes_per_token_forensic_ceiling':256},
}
DEFAULT_PR_PROFILE=CI_PROFILES['pr']
