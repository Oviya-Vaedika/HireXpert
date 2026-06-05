import streamlit as st
import google.generativeai as genai
import pypdf 
import docx
import re

# 🔑 Load Gemini API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# 🤖 Load model
model = genai.GenerativeModel("gemini-2.5-flash")

# 🎨 Page Config
st.set_page_config(page_title="HireXpert", page_icon="🤖", layout="wide")

st.title("🤖 HireXpert - Resume Analyzer")

# 📄 Function to extract text cleanly
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
            # Added a space separation to keep words from sticking together
            text = "\n".join([para.text for para in doc.paragraphs if para.text])

        elif uploaded_file.type == "text/plain":
            uploaded_file.seek(0) 
            text = uploaded_file.read().decode("utf-8")

        else:
            return None

    except Exception as e:
        # Logs the actual error to your Streamlit terminal for debugging
        print(f"Extraction error: {e}") 
        return None

    return text.strip()


# 📊 Fixed ATS Score Function
def calculate_score(resume, job_desc):
    if not resume or not job_desc:
        return 0.0
        
    # Convert text to lowercase sets of unique alphanumeric words
    resume_words = set(re.findall(r'\b\w+\b', resume.lower()))
    jd_words = set(re.findall(r'\b\w+\b', job_desc.lower()))
    
    # Filter out common filler words (stopwords)
    stop_words = {'and', 'the', 'is', 'in', 'at', 'of', 'for', 'with', 'a', 'to', 'an', 'on', 'or', 'by', 'be', 'from', 'i', 'am', 'are'}
    jd_keywords = jd_words - stop_words
    
    if not jd_keywords:
        return 0.0
        
    # Calculate how many critical JD keywords exist in the resume
    matching_words = jd_keywords.intersection(resume_words)
    score = (len(matching_words) / len(jd_keywords)) * 100
    
    return round(score, 2)


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

        # CRITICAL FIX: Use 'st.stop()' to halt execution if text extraction fails
        if not resume_text:
            st.error("❌ This file is empty, corrupted, or in an unsupported format.")
            st.stop() 
            
        # 📊 Score
        score = calculate_score(resume_text, job_desc)

        st.subheader("📊 ATS Score")
        # Ensure the progress bar receives a valid float between 0.0 and 1.0
        st.progress(score / 100.0)
            
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
