import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# --- 1. Page Configuration ---
st.set_page_config(page_title="HireXpert AI", page_icon="🎯", layout="wide")
st.title("HireXpert: Kaggle-Powered ATS 🤖")

# --- 2. Load Kaggle Data ---
@st.cache_data
def load_data():
    try:
        # This reads the Resume.csv file you just uploaded
        df = pd.read_csv("Resume.csv")
        # We group the data to make the AI smarter
        library = df.groupby('Category')['Resume_str'].apply(lambda x: ' '.join(x)).to_dict()
        return library
    except Exception as e:
        # Backup if file is missing
        return {"TEACHER": "education pedagogy teaching", "HEALTHCARE": "medical clinical doctor"}

JOB_LIBRARY = load_data()

# --- 3. Helper Functions ---
def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    text = ""
    if ext == 'pdf':
        pdf = PdfReader(file)
        text = " ".join([p.extract_text() for p in pdf.pages if p.extract_text()])
    elif ext in ['doc', 'docx']:
        doc = Document(io.BytesIO(file.read()))
        text = " ".join([para.text for para in doc.paragraphs])
    return text.strip()

def calculate_match(t1, t2):
    if not t1 or not t2: return 0.0
    tfidf = TfidfVectorizer(stop_words='english')
    vectors = tfidf.fit_transform([t1.lower(), t2.lower()])
    return round(cosine_similarity(vectors) * 100, 2)

# --- 4. Website Interface ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Job Description")
    job_desc = st.text_area("Paste the requirements here:", height=250)

with col2:
    st.subheader("📄 Upload Resume")
    uploaded_file = st.file_uploader("Upload PDF or Word", type=["pdf", "docx"])

st.divider()

if st.button("Run HireXpert Analysis"):
    if uploaded_file and job_desc.strip():
        with st.spinner("HireXpert AI is analyzing your profile..."):
            resume_text = extract_text(uploaded_file)
            
            # 1. Check if it's a resume
            if len(resume_text) < 100:
                st.error("❌ This file doesn't look like a proper resume. Please check the content.")
            else:
                # 2. Calculate ATS Match Score
                score = calculate_match(resume_text, job_desc)
                st.header(f"ATS Match Score: {score}%")
                
                # 3. Eligibility & Suggestions
                if score >= 50:
                    st.success("✅ **ELIGIBLE:** Great match for this role!")
                else:
                    st.warning("⚠️ **LOW MATCH:** Don't worry! Rejection is just redirection to something better. ✨")
                    st.write("---")
                    st.subheader("💡 Career Suggestions from Kaggle Database")
                    st.write("Our AI scanned thousands of resumes and found you are a better fit for:")
                    
                    # Compare resume against the 24+ categories in the CSV
                    suggestions = []
                    for job, desc in JOB_LIBRARY.items():
                        s = calculate_match(resume_text, desc)
                        suggestions.append((job, s))
                    
                    # Sort and show top 3
                    suggestions.sort(key=lambda x: x[1], reverse=True)
                    for job_name, job_score in suggestions[:3]:
                        st.write(f"- **{job_name}**: {job_score}% Match")
    else:
        st.error("Please paste a job description and upload a resume.")

st.sidebar.info(f"Connected to Kaggle: {len(JOB_LIBRARY)} industries loaded.")
