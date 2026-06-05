import streamlit as st
import google.generativeai as genai
import pypdf 
import docx
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔑 Load Gemini API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# 🤖 Load model
model = genai.GenerativeModel("gemini-2.5-flash")

# 🎨 Page Config
st.set_page_config(page_title="HireXpert", page_icon="🤖", layout="wide")

st.title("🤖 HireXpert - Resume Analyzer")

# 📄 Clean text extraction wrapper
def extract_text(uploaded_file):
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = pypdf.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs if para.text])

        elif uploaded_file.type == "text/plain":
            uploaded_file.seek(0) 
            text = uploaded_file.read().decode("utf-8")
        else:
            return None
    except Exception as e:
        print(f"Extraction error: {e}") 
        return None

    return text.strip()


# 📊 FIXED ATS SCORE: Uses Cosine Similarity for semantic/acronym overlap matching
def calculate_score(resume, job_desc):
    if not resume.strip() or not job_desc.strip():
        return 0.0
        
    try:
        # Vectorize using TF-IDF to accurately capture complex strings and professional terms
        vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
        tfidf_matrix = vectorizer.fit_transform([resume, job_desc])
        
        # Determine mathematical distance vector match
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Normalize and scale match up to a percentage representation
        score = similarity * 100
        return round(score, 2)
    except:
        return 0.0


# 📥 File Upload
resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

# 📝 Job Description
job_desc = st.text_area("Paste Job Description")

# 🚀 Analyze
if st.button("Analyze Resume"):

    if resume_file is None or job_desc.strip() == "":
        st.warning("Please upload resume and enter job description")
    else:
        resume_text = extract_text(resume_file)

        if not resume_text:
            st.error("❌ This file is empty, corrupted, or in an unsupported format.")
            st.stop() 
            
        # 📊 Calculate Semantic Vector Distance Match
        score = calculate_score(resume_text, job_desc)

        st.subheader("📊 ATS Score")
        st.progress(score / 100.0)
        st.write(f"**Score: {score}%**")

        # 🤖 AI Prompt injected with concrete context injection parameters
        current_date_str = datetime.now().strftime("%B %Y")
        prompt = f"""
        You are an expert technical recruiter analyzing a resume against a target Job Description.
        
        CRITICAL CONTEXT INDUCTION: 
        - The current actual date is exactly {current_date_str}. 
        - Do NOT treat dates matching or preceding {current_date_str} (such as March 2026) as future dates or hallucinated entries. 
        - A candidate list entry stating they are working "Till Now", "Present", or up to the current month signifies unbroken active tenure. Treat this as a profound operational strength.

        Please analyze this profile comprehensively:
        - Strengths (Include their tenure, active standing, and tier-1 vendor placements like HCL if present)
        - Weaknesses
        - Missing keywords
        - Improvement suggestions

        Resume Content:
        {resume_text}

        Job Description Requirements:
        {job_desc}
        """

        with st.spinner("Analyzing with AI..."):
            response = model.generate_content(prompt)
            st.subheader("🤖 AI Feedback")
            st.write(response.text)

        # 📥 Download Report
        report = f"""
ATS Score: {score}%

AI Feedback:
{response.text}
"""

        st.download_button(
            "📥 Download Report",
            report,
            file_name="resume_analysis.txt"
        )

#  Footer
st.markdown("---")
st.markdown("Made with ❤️ by a student")
