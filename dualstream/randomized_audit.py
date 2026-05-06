import hashlib, random

def randomized_selection(nonce:int|None,total_heads:int=16,subset_size:int=4)->dict:
    if nonce is None: nonce=0
    rng=random.Random(nonce)
    heads=sorted(rng.sample(list(range(total_heads)),k=min(subset_size,total_heads)))
    path=f'path-{nonce % 7}'
    nonce_hash=hashlib.sha256(str(nonce).encode()).hexdigest()[:16]
    return {'audit_nonce_hash':nonce_hash,'audit_path_id':path,'randomized_probe_selection':{'heads':heads,'subset_size':subset_size}}
