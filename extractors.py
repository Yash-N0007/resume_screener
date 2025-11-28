import re
import spacy
from pathlib import Path
from rapidfuzz import process, fuzz

nlp = spacy.load("en_core_web_sm")


def extract_name(raw: str, filename: str) -> str:
    """Extract candidate name."""
    doc = nlp(raw)
    people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if people:
        return people[0]
    name = Path(filename).stem.replace("_", " ").replace("-", " ")
    return re.sub(r"(?i)\b(cv|resume|final|2026)\b", "", name).strip()


def match_skills(text, required, cutoff=85):
    """Find required skills with fuzzy and whole-word matching."""
    found = []
    txt_lower = text.lower()
    for skill in required:
        s = skill.lower().strip()
        if not s:
            continue
        if re.search(rf"\b{s}\b", txt_lower):
            found.append(skill)
            continue
        if fuzz.partial_ratio(s, txt_lower) >= cutoff:
            found.append(skill)
    cov = 100 * len(found) / max(1, len(required))
    return sorted(set(found)), cov


import re

def extract_entities(text: str) -> dict:
    text_lower = text.lower()

    # Focus only on certifications and publications
    cert_patterns = re.findall(
        r"(?:certified(?:\s+in)?|certificate|certification|course|training|publication)[^.\n]{0,100}",
        text_lower
    )

    certificates = "; ".join(sorted(set([c.strip(" -•\n") for c in cert_patterns])))

    return {
        "certificates_found": certificates,
        "achievements_found": "",   # intentionally blank
        "competitions_won": ""      # intentionally blank
    }

# import re
# import spacy
# from pathlib import Path
# from rapidfuzz import fuzz

# nlp = spacy.load("en_core_web_sm")

# # -------- Name --------
# def extract_name(raw: str, filename: str) -> str:
#     doc = nlp(raw)
#     people = [e.text for e in doc.ents if e.label_=="PERSON"]
#     if people: return people[0]
#     name = Path(filename).stem.replace("_"," ").replace("-"," ")
#     return re.sub(r"(?i)\b(cv|resume|final|2026)\b","",name).strip()

# # -------- Skills (boundary + fuzzy) --------
# def match_skills(text: str, required: list[str], cutoff: int = 85):
#     found, txt = [], text.lower()
#     for skill in required:
#         s = (skill or "").lower().strip()
#         if not s: continue
#         if re.search(rf"\b{s}\b", txt):  # whole phrase
#             found.append(skill); continue
#         if fuzz.partial_ratio(s, txt) >= cutoff:  # near-phrase
#             # guard false positives like 'unity' in 'community'
#             if not re.search(rf"\b{s}\b", txt):  # if exact present it was already caught
#                 # small guard: skip if s is <=4 chars and only appears as substring
#                 if len(s) <= 4: continue
#             found.append(skill)
#     cov = round(100*len(set(found))/max(1,len(required)),1)
#     return sorted(set(found)), cov

# # -------- Sections + context --------
# SECTION_PATTERNS = {
#     "ach":  r"(?:Achievements?|Honours?|Honors?|Awards?|Leadership|Positions of Responsibility)",
#     "cert": r"(?:Certifications?|Courses?|Trainings?|Workshops?|Publications?)",
#     "proj": r"(?:Projects?|Experience|Internships?)",
# }

# def find_section(text: str, key: str, max_chars: int = 900) -> str:
#     pat = re.compile(rf"{SECTION_PATTERNS[key]}[\s:\-]*([\s\S]{{0,{max_chars}}})(?=\n[A-Z][^\n]*:|\Z)", re.IGNORECASE)
#     m = pat.search(text)
#     return m.group(1).strip() if m else ""

# def extract_entities(raw_text: str) -> dict:
#     # isolate sections
#     ach_block  = find_section(raw_text, "ach")
#     cert_block = find_section(raw_text, "cert")
#     proj_block = find_section(raw_text, "proj")

#     # regex triggers
#     ach_hits = re.findall(
#         r"\b(?:finalist|winner|runner[\s\-]?up|awarded|recognized|ranked|headed|led|organizer|organised|organized|mentor|speaker|presented|captain|chair|secretary)[^.\n]{0,120}",
#         ach_block, flags=re.IGNORECASE)
#     comp_hits = re.findall(
#         r"\b(?:hackathon|competition|contest|challenge|olympiad|tournament)[^.\n]{0,100}",
#         ach_block, flags=re.IGNORECASE)
#     cert_hits = re.findall(
#         r"\b(?:certified(?:\s+in)?|certificate|certification|course|training|workshop|publication)[^.\n]{0,140}",
#         cert_block, flags=re.IGNORECASE)
#     proj_hits = re.findall(
#         r"(?:project|experience|internship|developed|built|designed|implemented)[^.\n]{0,180}",
#         proj_block or raw_text, flags=re.IGNORECASE)

#     # NER augmentation
#     doc = nlp(" ".join([ach_block, cert_block, proj_block]))
#     ner_items = [ent.text.strip() for ent in doc.ents if ent.label_ in ("ORG","EVENT","WORK_OF_ART","PRODUCT","TITLE")]

#     def clean(items): return [re.sub(r"[-•;:\s]+$", "", i.strip()) for i in items if i and len(i.strip())>3]
#     def join(items):  return "; ".join(sorted(set(clean(items))))

#     certificates = join(cert_hits + [x for x in ner_items if len(x.split())>1])
#     achievements = join(ach_hits + [x for x in ner_items if len(x.split())>1])
#     competitions = join(comp_hits)
#     projects     = join(proj_hits)

#     # unified, short context for LLM
#     context_excerpt = "; ".join(filter(None,[projects, achievements, certificates, competitions]))
#     context_excerpt = re.sub(r"\s{2,}"," ", context_excerpt)[:1500]

#     return {
#         "certificates_found": certificates,
#         "achievements_found": achievements,
#         "competitions_won": competitions,
#         "projects_found": projects,
#         "context_excerpt": context_excerpt,
#     }
