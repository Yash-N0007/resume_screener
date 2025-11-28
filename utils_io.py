from pathlib import Path
from pdfminer.high_level import extract_text
import docx, re

def read_text(path: Path) -> str:
    if path.suffix.lower()==".pdf": return extract_text(str(path))
    if path.suffix.lower()==".docx": return "\n".join(p.text for p in docx.Document(path).paragraphs)
    return path.read_text(errors="ignore")

def normalize(t: str) -> str:
    t = re.sub(r"[^\x00-\x7F]+"," ",t)
    t = re.sub(r"\s+"," ",t).strip().lower()
    return t
