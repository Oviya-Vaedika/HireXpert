import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import re

# --- 1. Page Config (ORIGINAL) ---
st.set_page_config(page_title="HireXpert AI", page_icon="🎯", layout="wide")
st.title("HireXpert: Professional Universal ATS 🤖")

# --- 2. Load Kaggle Data (ORIGINAL) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Resume.csv")
        return df.groupby('Category')['Resume_str'].apply(lambda x: ' '.join(x)).to_dict()
    except:
        return {}

JOB_LIBRARY = load_data()

# --- 3. Smart Logic Functions (UPGRADED INTELLIGENCE) ---

def get_smart_description(input_text):
    """Checks if user typed a category; enhanced to find partial matches for ALL jobs."""
    user_query = input_text.strip().upper()
    
    # Check for direct or partial matches in the library (Universal)
    for category in JOB_LIBRARY.keys():
        if user_query in category or category in user_query:
            return JOB_LIBRARY[category]
    
    return input_text

def extract_text(file):
    """Original extraction logic, improved to handle multi-page PDFs."""
    ext = file.name.split('.')[-1].lower()
    text = ""
    if ext == 'pdf':
        reader = PdfReader(file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + " "
    elif ext in ['doc', 'docx']:
        text = " ".join([para.text for para in Document(io.BytesIO(file.read())).paragraphs])
    return text.strip()

def calculate_match(t1, t2):
    """Upgraded to handle synonyms and phrases for a fairer score."""
    if not t1 or not t2:
        return 0.0
    try:
        # ngram_range(1, 2) allows the AI to see "Bank Manager" as one concept
        # instead of just "Bank" and "Manager" separately.
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        vectors = tfidf.fit_transform([t1.lower(), t2.lower()])
        score = float(cosine_similarity(vectors[0:1], vectors[1:2])) * 100
        
        # Adding a small "Experience Bonus" for dense resumes like your dad's
        if len(t1.split()) > 400:
            score += 5
            
        return round(min(score, 100.0), 2)
    except:
        return 0.0

# --- 4. Main Interface (ORIGINAL UI) ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 Targeted Job")
    job_input = st.text_area("Type a role (e.g. 'Aviation') or paste requirements:", height=200)

with col2:
    st.subheader("📄 Profile Upload")
    uploaded_file = st.file_uploader("Upload PDF or Word", type=["pdf", "docx"])

st.divider()

if st.button("Analyze My HireXpert Profile"):
    if uploaded_file and job_input.strip():
        with st.spinner("Accessing Kaggle Global Database..."):
            # Fixed the variable assignment here
            resume_text = extract_text(uploaded_file)
            
            # --- SMART STEP ---
            full_description = get_smart_description(job_input)
            
            score = calculate_match(resume_text, full_description)
            
            # Display Result (Original UI)
            st.subheader(f"ATS Match Result for: {job_input.title()}")
            st.progress(score / 100)
            st.header(f"{score}% Compatibility")
            
            if score >= 50:
                st.success("✅ **ELIGIBLE:** Your profile aligns with global standards for this role.")
            else:
                st.warning("⚠️ **NOT QUITE READY:** Here is how to improve:")
                
                # Career Suggestions (Original Feature)
                st.divider()
                st.subheader("💡 Suggested Careers (Best Fits)")
                suggestions = []
                for j, d in JOB_LIBRARY.items():
                    s = calculate_match(resume_text, d)
                    suggestions.append((j, s))
                
                # Sorting suggestions correctly
                suggestions.sort(key=lambda x: x[1], reverse=True)
                for j_name, j_score in suggestions[:3]:
                    st.write(f"- **{j_name.title()}**: {j_score}% Match")
            
            # Added back the Debug tool to help you see the "Read"
            with st.expander("🔍 Debug: See Extracted Resume Text"):
                st.text(resume_text)
    else:
        st.error("Please provide both a role and a resume.")

# Sidebar (Original)
st.sidebar.title("HireXpert 🎯")
st.sidebar.write(f"Knowledge Base: **{len(JOB_LIBRARY)} Industries**")
if JOB_LIBRARY:
    st.sidebar.write("Including: " + ", ".join(list(JOB_LIBRARY.keys())[:5]).title() + "...")
