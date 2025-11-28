import gradio as gr
from core import evaluate_folder
from pathlib import Path
import tempfile, shutil

def run_screening(job_description, skills_text, files):
    if not files: return "No files uploaded.", None
    tmpdir = Path(tempfile.mkdtemp())
    for f in files: shutil.copy(f.name, tmpdir / Path(f.name).name)
    required_skills = [s.strip().lower() for s in skills_text.split(",") if s.strip()]
    df = evaluate_folder(tmpdir, job_description, required_skills)
    csv_path = Path("outputs/hybrid_results.csv")
    df.to_csv(csv_path, index=False)
    return df, str(csv_path)

iface = gr.Interface(
    fn=run_screening,
    inputs=[
        gr.Textbox(label="Job Description", lines=6),
        gr.Textbox(label="Required Skills (comma separated)", placeholder="python, sql, pandas, machine learning"),
        gr.File(label="Upload resumes (PDF/DOCX/TXT)", file_count="multiple", type="filepath")
    ],
    outputs=[gr.Dataframe(label="Results", interactive=False),
             gr.File(label="Download CSV")],
    title="Hybrid Resume Screener",
    description="Rules + SentenceTransformers + Llama3 with section-aware context"
)

if __name__ == "__main__":
    iface.launch()
