import json, re, datetime
from typing import Dict, List
import ollama

LOG_PATH = "outputs/llama_logs.txt"
MODEL = "llama3"

def _log(data: dict):
    try:
        with open(LOG_PATH,"a",encoding="utf-8") as f:
            f.write(f"\n----- {datetime.datetime.now()} -----\n")
            f.write(json.dumps(data, indent=2)); f.write("\n")
    except Exception: pass

def rule_features(found_skills: List[str], entities: Dict[str,str], required_skills: List[str], bi: float, ce: float) -> Dict[str,str]:
    overlap = set(map(str.lower, found_skills)).intersection(set(map(str.lower, required_skills)))
    coverage = round(100*len(overlap)/max(1,len(required_skills)),1)
    base = "Strong" if len(overlap)>=4 else "Moderate" if len(overlap)>=2 else "Weak"
    return {
        "matching_skills": ", ".join(sorted(overlap)),
        "skills_coverage_%": coverage,
        "base_assessment": base,
        "bi_encoder_%": round(bi,2),
        "cross_encoder_%": round(ce,2),
    }

def llama_refine(job_description: str, rule_data: dict, rag_context: str) -> dict:
    payload = {
        "job_description": job_description,
        "rule_features": rule_data,
        "rag_context": rag_context[:1500],
    }

    _log({"backend": "ollama", "payload": payload})
    prompt = f"""
You are an HR technical screener. 
Your goal is to judge **role suitability** based ONLY on explicit evidence in the resume.
Do NOT make analogies or guesses.

Input data:
{json.dumps(payload, indent=2)}

Follow these rules:
1. Read the job description literally. Identify its key technical requirements.
2. Look for direct evidence in the resume text or context. 
   - Programming languages, frameworks, projects, or courses.
   - Publications or certifications relevant to the role.
3. If the role and resume belong to different domains (e.g., Data Science vs UI/UX Design), 
   mark Role_Fit as "No".
4. Keep reasoning short, factual, and evidence-based.

Return STRICT JSON:
{{
  "Role_Fit": "Yes" or "No",
  "Reasoning": "1–2 line factual justification",
  "Refined_Certifications": "comma-separated factual certifications/publications",
  "Recommended_Skills_To_Add": "comma-separated list of missing but relevant technical skills"
}}
"""

# V1
#     prompt = f"""
# You are a professional technical recruiter.
# Use the provided structured data and resume excerpt to assess candidate fit.
# Return STRICT JSON only in this exact schema:

# {{
#   "Role_Fit": "Yes" or "No",
#   "Reasoning": "short factual summary (2 lines)",
#   "Refined_Certifications": "summarized list of certifications or publications only",
#   "Recommended_Skills_To_Add": "comma-separated list of missing but relevant skills"
# }}

# DATA:
# {json.dumps(payload, indent=2)}

# Notes:
# - Only recommend *skills* that align with the job description or are absent from the candidate's current skill set.
# - Do NOT include personality traits or soft skills.
# """
    resp = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    m = re.search(r"\{[\s\S]*\}", resp["message"]["content"])
    return json.loads(m.group(0)) if m else {}


def intent_reasoner(job_description, found_skills, entities, required_skills, bi, ce, rag_context):
    rules = rule_features(found_skills, entities, required_skills, bi, ce)
    llm = llama_refine(job_description, rules, rag_context)
    return {**rules, **llm}

