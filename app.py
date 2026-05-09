import streamlit as st
import google.generativeai as genai
import PyPDF2
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔑 Load Gemini API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# 🤖 Load model
model = genai.GenerativeModel("gemini-1.5-flash")

# 🎨 Page Config
st.set_page_config(page_title="HireXpert", page_icon="🤖", layout="wide")

st.title("🤖 HireXpert - Resume Analyzer")

# 📄 Function to extract text
def extract_text(uploaded_file):
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text()

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text

        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")

        else:
            return None

    except:
        return None

    return text


# 📊 ATS Score Function
def calculate_score(resume, job_desc):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume, job_desc])
    score = cosine_similarity(vectors[0:1], vectors[1:2])
    return round(score[0][0] * 100, 2)


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
            st.error("❌ This file is not a valid resume or unsupported format")
        else:
            # 📊 Score
            score = calculate_score(resume_text, job_desc)

            st.subheader("📊 ATS Score")
            st.progress(int(score))
            st.write(f"**Score: {score}%**")

            # 🤖 AI Feedback
            prompt = f"""
            Analyze this resume and give:
            - Strengths
            - Weaknesses
            - Missing keywords
            - Improvement suggestions

            Resume:
            {resume_text}

            Job Description:
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