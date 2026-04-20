import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# --- 1. Page Config ---
st.set_page_config(page_title="HireXpert AI", page_icon="🎯", layout="wide")
st.title("HireXpert: Professional Universal ATS 🤖")

# --- 2. Load Kaggle Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Resume.csv")
        # Ensure categories are uppercase and stripped of spaces for better matching
        df['Category'] = df['Category'].str.upper().str.strip()
        return df.groupby('Category')['Resume_str'].apply(lambda x: ' '.join(x)).to_dict()
    except Exception as e:
        st.error(f"Error loading Resume.csv: {e}")
        return {}

JOB_LIBRARY = load_data()

# --- 3. Improved Logic Functions ---

def get_smart_description(input_text):
    """Matches user input to the database more flexibly."""
    user_query = input_text.strip().upper()
    
    # Check for direct or keyword matches (e.g., 'Bank' matching 'BANKING')
    for category in JOB_LIBRARY.keys():
        if user_query in category or category in user_query or "BANK" in user_query:
            return JOB_LIBRARY[category]
    
    return input_text

def extract_text(file):
    """Extracts text from PDF or DOCX."""
    text = ""
    try:
        ext = file.name.split('.')[-1].lower()
        if ext == 'pdf':
            reader = PdfReader(file)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content + " "
        elif ext in ['doc', 'docx']:
            doc = Document(io.BytesIO(file.read()))
            for para in doc.paragraphs:
                text += para.text + " "
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return text.strip()

def calculate_match(resume_text, job_text):
    """Calculates TF-IDF similarity score."""
    if not resume_text or not job_text:
        return 0.0
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        # Combine texts to create a shared vocabulary
        vectors = tfidf.fit_transform([resume_text.lower(), job_text.lower()])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return round(float(similarity) * 100, 2)
    except:
        return 0.0

# --- 4. Main Interface ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 Targeted Job")
    job_input = st.text_area("Type a role (e.g. 'Banking') or paste requirements:", 
                             placeholder="Try 'Banking' or 'Bank Manager'", height=200)

with col2:
    st.subheader("📄 Profile Upload")
    uploaded_file = st.file_uploader("Upload PDF or Word", type=["pdf", "docx"])

st.divider()

if st.button("Analyze My HireXpert Profile"):
    if uploaded_file and job_input.strip():
        with st.spinner("Analyzing profile against global database..."):
            # 1. Extract text from the uploaded file
            resume_content = extract_text(uploaded_file)
            
            # 2. Get the comparison text (Smart lookup)
            comparison_text = get_smart_description(job_input)
            
            # 3. Calculate Score
            score = calculate_match(resume_content, comparison_text)
            
            # --- Display Results ---
            st.subheader(f"ATS Match Result for: {job_input.title()}")
            st.progress(score / 100)
            
            if score > 0:
                st.header(f"{score}% Compatibility")
            else:
                st.header("0.0% Compatibility")
                st.error("No overlap found. Ensure the resume text is readable and the job title matches a known industry.")

            if score >= 50:
                st.success("✅ **ELIGIBLE:** This profile aligns well with industry standards.")
            else:
                st.warning("⚠️ **IMPROVEMENT NEEDED:** See suggestions below.")
                
                # Career Suggestions
                st.divider()
                st.subheader("💡 Top Industry Fits for This Resume")
                suggestions = []
                for category_name, category_text in JOB_LIBRARY.items():
                    sim_score = calculate_match(resume_content, category_text)
                    suggestions.append((category_name, sim_score))
                
                # Sort by highest score
                suggestions.sort(key=lambda x: x[1], reverse=True)
                
                for j_name, j_score in suggestions[:3]:
                    st.write(f"- **{j_name.title()}**: {j_score}% Match")

            # --- DEBUG SECTION (Optional) ---
            with st.expander("🔍 Debug: See Extracted Resume Text"):
                if resume_content:
                    st.write(resume_content[:1000] + "...") 
                else:
                    st.write("No text could be extracted from the file.")

    else:
        st.error("Please provide both a job title and a resume file.")

# Sidebar
st.sidebar.title("HireXpert 🎯")
st.sidebar.write(f"Knowledge Base: **{len(JOB_LIBRARY)} Industries**")
if JOB_LIBRARY:
    st.sidebar.write("Available Categories:")
    st.sidebar.code(", ".join(list(JOB_LIBRARY.keys())[:10]))
