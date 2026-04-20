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

# --- 2. Load Kaggle Data (FIXED LOADING) ---
@st.cache_data
def load_data():
    try:
        # Ensure the filename matches exactly
        df = pd.read_csv("Resume.csv")
        # Standardize categories and group them
        df['Category'] = df['Category'].str.upper().str.strip()
        return df.groupby('Category')['Resume_str'].apply(lambda x: ' '.join(str(i) for i in x)).to_dict()
    except Exception as e:
        st.sidebar.error(f"Could not load Resume.csv: {e}")
        return {}

JOB_LIBRARY = load_data()

# --- 3. Smart Logic Functions ---

def get_smart_description(input_text):
    user_query = input_text.strip().upper()
    for category in JOB_LIBRARY.keys():
        if user_query in category or category in user_query:
            return JOB_LIBRARY[category]
    return input_text

def extract_text(file):
    text = ""
    try:
        ext = file.name.split('.')[-1].lower()
        if ext == 'pdf':
            reader = PdfReader(file)
            for page in reader.pages:
                content = page.extract_text()
                if content: text += content + " "
        elif ext in ['doc', 'docx']:
            text = " ".join([para.text for para in Document(io.BytesIO(file.read())).paragraphs])
    except:
        pass
    return text.strip()

def calculate_match(t1, t2):
    if not t1 or not t2:
        return 0.0
    try:
        # ngram_range(1,2) helps match phrases like "Bank Manager"
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        vectors = tfidf.fit_transform([t1.lower(), t2.lower()])
        score = float(cosine_similarity(vectors[0:1], vectors[1:2])) * 100
        return round(score, 2)
    except:
        return 0.0

# --- 4. Main Interface ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 Targeted Job")
    job_input = st.text_area("Type a role or paste requirements:", height=200)

with col2:
    st.subheader("📄 Profile Upload")
    uploaded_file = st.file_uploader("Upload PDF or Word", type=["pdf", "docx"])

st.divider()

if st.button("Analyze My HireXpert Profile"):
    if uploaded_file and job_input.strip():
        with st.spinner("Analyzing..."):
            resume_text = extract_text(uploaded_file)
            full_description = get_smart_description(job_input)
            
            score = calculate_match(resume_text, full_description)
            
            # Display Result
            st.subheader(f"ATS Match Result for: {job_input.title()}")
            st.progress(score / 100)
            st.header(f"{score}% Compatibility")
            
            # --- FIXED CAREER SUGGESTIONS ---
            st.divider()
            st.subheader("💡 Suggested Careers (Best Fits)")
            
            if JOB_LIBRARY:
                suggestions = []
                for category_name, category_data in JOB_LIBRARY.items():
                    # Calculate similarity for each industry
                    sim_score = calculate_match(resume_text, category_data)
                    suggestions.append((category_name, sim_score))
                
                # Sort by highest score first
                suggestions.sort(key=lambda x: x[1], reverse=True)
                
                # Show top 5 matches
                for j_name, j_score in suggestions[:5]:
                    st.write(f"**{j_name.title()}**: {j_score}% Match")
            else:
                st.error("Knowledge base is empty. Please check if Resume.csv is in the folder.")

            # Optional Debug
            with st.expander("🔍 Debug: See Extracted Resume Text"):
                st.text(resume_text)
    else:
        st.error("Please provide both a role and a resume.")

# Sidebar
st.sidebar.title("HireXpert 🎯")
st.sidebar.write(f"Knowledge Base: **{len(JOB_LIBRARY)} Industries**")
