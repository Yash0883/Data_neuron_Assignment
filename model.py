from sentence_transformers import SentenceTransformer, util
import torch

# Force CPU to avoid GPU-related issues on Render (which uses CPU by default)
device = "cpu"
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts using SBERT embeddings.
    Returns a value between 0 and 1 (rounded to 3 decimals).
    """
    # ensure strings
    t1 = "" if text1 is None else str(text1)
    t2 = "" if text2 is None else str(text2)

    emb1 = model.encode(t1, convert_to_tensor=True)
    emb2 = model.encode(t2, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(emb1, emb2).item()
    score = (cosine_sim + 1.0) / 2.0  # [-1,1] -> [0,1]
    # clamp and round
    score = max(0.0, min(1.0, score))
    return float(round(score, 3))
