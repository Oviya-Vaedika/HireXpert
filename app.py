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

# --- 2. Load Kaggle Data (Universal) ---
@st.cache_data
def load_data():
    try:
        # This reads the Resume.csv you uploaded
        df = pd.read_csv("Resume.csv")
        # This groups ALL categories from the Kaggle file (24+ industries)
        # It automatically includes Accountant, Advocate, Agriculture, Aviation, Banking, etc.
        return df.groupby('Category')['Resume_str'].apply(lambda x: ' '.join(x)).to_dict()
    except:
        return {}

JOB_LIBRARY = load_data()

# --- 3. Smart Logic Functions ---

def get_smart_description(input_text):
    """Checks if the user typed a known category from the Kaggle data."""
    # Convert user input to uppercase to match Kaggle categories (e.g. 'doctor' -> 'HEALTHCARE')
    user_query = input_text.strip().upper()
    
    # Check for direct or partial matches in the Kaggle categories
    for category in JOB_LIBRARY.keys():
        if user_query in category or category in user_query:
            return JOB_LIBRARY[category]
    
    # If no match is found, just use the text the user typed
    return input_text

def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return " ".join([p.extract_text() for p in PdfReader(file).pages if p.extract_text()])
    elif ext in ['doc', 'docx']:
        return " ".join([para.text for para in Document(io.BytesIO(file.read())).paragraphs])
    return ""

def calculate_match(t1, t2):
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        vectors = tfidf.fit_transform([t1.lower(), t2.lower()])
        return round(cosine_similarity(vectors) * 100, 2)
    except:
        return 0.0

# --- 4. Main Interface ---
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
            resume_text = extract_text(uploaded_resume_file := uploaded_file)
            
            # --- SMART STEP ---
            # If they type 'Aviation', it finds the Aviation data in the CSV
            full_description = get_smart_description(job_input)
            
            score = calculate_match(resume_text, full_description)
            
            # Display Result
            st.subheader(f"ATS Match Result for: {job_input.title()}")
            st.progress(score / 100)
            st.header(f"{score}% Compatibility")
            
            if score >= 50:
                st.success("✅ **ELIGIBLE:** Your profile aligns with global standards for this role.")
            else:
                st.warning("⚠️ **NOT QUITE READY:** Here is how to improve:")
                
                # Career Suggestions
                st.divider()
                st.subheader("💡 Suggested Careers (Best Fits)")
                suggestions = []
                for j, d in JOB_LIBRARY.items():
                    s = calculate_match(resume_text, d)
                    suggestions.append((j, s))
                suggestions.sort(key=lambda x: x, reverse=True)
                for j_name, j_score in suggestions[:3]:
                    st.write(f"- **{j_name.title()}**: {j_score}% Match")
    else:
        st.error("Please provide both a role and a resume.")

# Sidebar
st.sidebar.title("HireXpert 🎯")
st.sidebar.write(f"Knowledge Base: **{len(JOB_LIBRARY)} Industries**")
if JOB_LIBRARY:
    st.sidebar.write("Including: " + ", ".join(list(JOB_LIBRARY.keys())[:5]).title() + "...")
