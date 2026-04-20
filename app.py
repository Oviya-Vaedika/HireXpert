import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# --- 1. Page Config ---
st.set_page_config(page_title="HireXpert AI", page_icon="🎯", layout="wide")
st.title("HireXpert: Professional ATS Optimizer 🤖")

# --- 2. Load Kaggle Data (Same as before) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Resume.csv")
        return df.groupby('Category')['Resume_str'].apply(lambda x: ' '.join(x)).to_dict()
    except:
        return {"TEACHER": "education", "HEALTHCARE": "medical"}

JOB_LIBRARY = load_data()

# --- 3. Extra Feature: Extract Missing Keywords ---
def get_missing_keywords(resume, job):
    resume_words = set(resume.lower().split())
    job_words = set(job.lower().split())
    # Find words in Job but not in Resume (ignoring common small words)
    stop_words = {'and', 'the', 'for', 'with', 'from', 'this', 'that', 'your', 'have'}
    missing = job_words - resume_words - stop_words
    return list(missing)[:10] # Top 10 missing words

# --- 4. Helper Functions ---
def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return " ".join([p.extract_text() for p in PdfReader(file).pages if p.extract_text()])
    elif ext in ['doc', 'docx']:
        return " ".join([para.text for para in Document(io.BytesIO(file.read())).paragraphs])
    return ""

def calculate_match(t1, t2):
    tfidf = TfidfVectorizer(stop_words='english')
    vectors = tfidf.fit_transform([t1.lower(), t2.lower()])
    return round(cosine_similarity(vectors) * 100, 2)

# --- 5. UI Layout ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 Job Description")
    job_desc = st.text_area("Paste Requirements:", height=200)
with col2:
    st.subheader("📄 Resume Upload")
    uploaded_file = st.file_uploader("PDF or Word", type=["pdf", "docx"])

if st.button("Generate HireXpert Report"):
    if uploaded_file and job_desc:
        resume_text = extract_text(uploaded_file)
        score = calculate_match(resume_text, job_desc)
        
        # FEATURE 2: Visual Progress Bar
        st.subheader("ATS Compatibility Result")
        st.progress(score / 100) # This creates a visual bar
        st.write(f"Your Match Score: **{score}%**")
        
        # FEATURE 1: Missing Keywords
        missing = get_missing_keywords(resume_text, job_desc)
        if missing:
            st.info(f"💡 **AI Tip:** Try adding these keywords to your resume: {', '.join(missing)}")

        if score < 50:
            st.divider()
            st.subheader("💡 Smart Career Suggestions")
            suggestions = []
            for j, d in JOB_LIBRARY.items():
                s = calculate_match(resume_text, d)
                suggestions.append((j, s))
            suggestions.sort(key=lambda x: x[1], reverse=True)
            for j_name, j_score in suggestions[:3]:
                st.write(f"- **{j_name}**: {j_score}% Match")
    else:
        st.error("Please provide both inputs.")

# Sidebar Branding
st.sidebar.title("HireXpert 🎯")
st.sidebar.markdown("Developing career paths using **Big Data**.")
