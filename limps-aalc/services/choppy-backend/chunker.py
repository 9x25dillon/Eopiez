import math

def chunk_text(text: str, max_tokens: int = 512, overlap: float = 0.15):
    words = text.split()
    step = int(max_tokens * (1-overlap))
    out = []
    i = 0
    while i < len(words):
        out.append(" ".join(words[i:i+max_tokens]))
        i += max(1, step)
    return out