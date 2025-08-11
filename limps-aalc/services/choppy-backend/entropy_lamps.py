import math

def token_entropy(s: str):
    # toy entropy estimator by char frequency
    if not s: return 0.0
    from collections import Counter
    c = Counter(s)
    n = sum(c.values())
    H = 0.0
    for v in c.values():
        p = v/n
        H -= p*math.log(p+1e-12, 2)
    return H

def entropy_report(chunks):
    vals = [token_entropy(c) for c in chunks]
    if not vals: return {"avg":0,"min":0,"max":0}
    return {"avg": sum(vals)/len(vals), "min": min(vals), "max": max(vals)}