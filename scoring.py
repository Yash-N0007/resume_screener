from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch

enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def bi_score(resume_text_norm: str, jd_text_norm: str) -> float:
    r = enc.encode([resume_text_norm], normalize_embeddings=True)
    j = enc.encode([jd_text_norm], normalize_embeddings=True)
    return float(util.cos_sim(r, j)[0][0]) * 100

def cross_score(jd_raw: str, resume_raw: str) -> float:
    raw = float(cross.predict([(jd_raw, resume_raw)])[0])
    return float(torch.sigmoid(torch.tensor(raw))) * 100
