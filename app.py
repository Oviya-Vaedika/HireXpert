import streamlit as st
import PyPDF2
import docx
import re
from openai import OpenAI

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="HireXpert AI", page_icon="🤖", layout="wide")

# -------------------------------
# OPENAI SETUP
# -------------------------------
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error(f"API Key Error: {e}")

# -------------------------------
# FUNCTIONS
# -------------------------------
def extract_text(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return "".join([page.extract_text() or "" for page in pdf_reader.pages])

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])

        else:
            return str(uploaded_file.read(), "utf-8")

    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return " ".join(text.split())

def get_ai_analysis(resume_text, jd_text):
    prompt = f"""
    Act as an expert HR Manager and ATS system.

    1. Start with Match Score (0-100%)
    2. Top 5 Missing Keywords
    3. Improvement Summary

    Job Description:
    {jd_text}

    Resume:
    {resume_text}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional ATS resume analyzer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI Error: {e}"

# -------------------------------
# 🎨 PREMIUM UI DESIGN
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1 {
    color: #ffffff;
    font-weight: 800;
}
.block-container {
    padding-top: 2rem;
}
.stButton>button {
    background: linear-gradient(90deg, #6c63ff, #00c9a7);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
}
.stTextArea textarea {
    border-radius: 10px;
}
.stFileUploader {
    border: 2px dashed #6c63ff;
    padding: 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
<h1>🚀 HireXpert AI</h1>
<p style='color:gray;'>Smart Resume Screening • ATS Score • Instant Feedback</p>
""", unsafe_allow_html=True)

st.divider()

# -------------------------------
# INPUTS
# -------------------------------
jd = st.text_area("📄 Paste Job Description", height=150)
uploaded_file = st.file_uploader("📤 Upload Resume", type=["pdf", "docx", "txt"])

# -------------------------------
# MAIN LOGIC
# -------------------------------
if uploaded_file and jd:
    resume_raw = extract_text(uploaded_file)

    if resume_raw:
        resume_text_clean = clean_text(resume_raw)

        st.markdown("### ⚡ Actions")
        col1, col2 = st.columns(2)

        # 🔍 ANALYSIS
        with col1:
            if st.button("🔍 Analyze Resume"):
                with st.spinner("Analyzing your resume..."):
                    result = get_ai_analysis(resume_text_clean, jd)
                    st.session_state["ai_report"] = result

                    st.markdown("## 📊 Results")
                    st.success("Analysis Complete!")
                    st.markdown(result)

        # ✨ TIPS
        with col2:
            if st.button("✨ Improve Resume"):
                st.markdown("## ✨ Suggestions")

                weak_words = {
                    "managed": "Spearheaded",
                    "helped": "Facilitated",
                    "led": "Orchestrated",
                    "worked": "Collaborated"
                }

                found = False
                for weak, strong in weak_words.items():
                    if weak in resume_text_clean:
                        st.write(f"🔁 {weak} → **{strong}**")
                        found = True

                if not found:
                    st.success("🔥 Your resume wording looks strong!")

        # 📥 DOWNLOAD
        if "ai_report" in st.session_state:
            st.download_button(
                "📥 Download Report",
                data=st.session_state["ai_report"],
                file_name="hirexpert_report.txt"
            )

    else:
        st.error("Could not read resume file.")

else:
    st.info("Upload resume + enter job description to start.")

# -------------------------------
# FOOTER
# -------------------------------
st.divider()
st.caption("Made with ❤️ by a student developer")