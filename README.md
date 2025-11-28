# Hybrid Resume Screening System  
A rules + embeddings + LLM–powered resume ranking system with skill coverage analysis, bi-encoder and cross-encoder similarity scores, RAG-based context extraction, and Llama-3 reasoning.

This tool provides an intelligent and explainable way to evaluate multiple resumes against a job description using NLP, embeddings, fuzzy matching, and LLM refinement.

---

# 🚀 Key Features

## 🔹 1. **Resume Parsing & Normalization**
- Extracts text from **PDF, DOCX, TXT** files  
  (using `pdfminer.six`, `python-docx`)  
- Normalizes casing, whitespace, and special characters  
  :contentReference[oaicite:0]{index=0}

## 🔹 2. **Skill Extraction & Coverage**
- Whole-word matching  
- Fuzzy matching with RapidFuzz (≥85 threshold)  
- Computes `% coverage` of required skills  
  :contentReference[oaicite:1]{index=1}

## 🔹 3. **Entity Extraction**
- Identifies **certifications, courses, publications**  
- Extracts structured info from resume text  
  :contentReference[oaicite:2]{index=2}

## 🔹 4. **Embedding-Based Scoring**
Two independent similarity engines:

### **Bi-Encoder Score**
- SentenceTransformer (`all-mpnet-base-v2`)
- Cosine similarity ×100  
  :contentReference[oaicite:3]{index=3}

### **Cross-Encoder Score**
- CrossEncoder (`ms-marco-MiniLM-L-6-v2`)
- Sigmoid activation for % relevance  
  :contentReference[oaicite:4]{index=4}

### **Final Weighted Score**
final_score = 0.3 * cross_encoder
+ 0.6 * bi_encoder
+ 0.1 * skills_coverage

- Predicts **YES/NO** match threshold at 50  
  :contentReference[oaicite:5]{index=5}

---

# 🔹 5. **RAG – Resume-Aware Context Retrieval**
Each resume is indexed into its **own isolated Qdrant vector collection**:

- Embeds resume text in 800-char chunks  
- Retrieves top-k relevant chunks for queries such as:  
  `"certifications or publications"`  
  :contentReference[oaicite:6]{index=6}

---

# 🔹 6. **LLM Reasoning (Llama-3 via Ollama)**
The system sends structured data to Llama-3:

- Required skills  
- Matching skills  
- Embedding scores  
- Extracted entities  
- Retrieved RAG context  

The LLM returns strict JSON with:

- **Role_Fit (Yes/No)**  
- **Reasoning (2-line justification)**  
- **Refined Certifications**  
- **Recommended Skills To Add**  
  :contentReference[oaicite:7]{index=7}

---

# 🔹 7. **Gradio Web App Interface**
A clean UI for uploading multiple resumes at once:

- Job description input  
- Required skills input  
- Multi-file upload  
- Displays results as a dataframe  
- Allows CSV download  
  :contentReference[oaicite:8]{index=8}

---


