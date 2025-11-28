from pathlib import Path
import pandas as pd
from utils_io import read_text, normalize
from extractors import extract_name, extract_entities, match_skills
from scoring import bi_score, cross_score
from intent_hybrid import intent_reasoner
from rag_store import index_resume, retrieve_context

SUPPORTED = {".pdf", ".docx", ".txt"}


def evaluate_folder(res_dir: Path, job_description: str, required_skills: list[str]) -> pd.DataFrame:
    Path("outputs").mkdir(exist_ok=True)
    rows = []
    jd_norm = normalize(job_description)

    for fp in Path(res_dir).iterdir():
        if fp.suffix.lower() not in SUPPORTED:
            continue

        raw = read_text(fp)
        txt = normalize(raw)
        name = extract_name(raw, fp.name)

        # Skills and entities
        found, cov = match_skills(txt, required_skills)
        entities = extract_entities(raw)

        # Embedding scores
        bi = bi_score(txt, jd_norm)
        ce = cross_score(job_description, raw)

        # Final weighted score
        final = round(0.3 * ce + 0.6 * bi + 0.1 * cov, 2)
        match = "YES" if final >= 50 else "NO"

        # --- RAG context isolation ---
        collection_name = index_resume(fp.name, raw)
        rag_context = retrieve_context(collection_name, "certifications or publications")

        # --- Intent reasoning (Llama + rules + retrieved context) ---
        intent = intent_reasoner(job_description, found, entities, required_skills, bi, ce, rag_context)

        # --- Append results ---
        rows.append({
            "name": name,
            "file": fp.name,
            "bi_encoder_%": round(bi, 2),
            "cross_encoder_%": round(ce, 2),
            "skills_coverage_%": round(cov, 1),
            "final_score_%": final,
            "predicted_match": match,
            **intent,  # includes Role_Fit, Reasoning, etc.
            "Refined_Certifications": intent.get("Refined_Certifications", ""),
            "Recommended_Skills_To_Add": intent.get("Recommended_Skills_To_Add", "")
        })

    df = pd.DataFrame(rows).sort_values(
        ["predicted_match", "cross_encoder_%", "bi_encoder_%"],
        ascending=[True, False, False]
    )
    df.to_csv("outputs/hybrid_results.csv", index=False)
    return df
