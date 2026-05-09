import streamlit as st
from google import genai
import PyPDF2
import docx
import re
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# 🔑 NEW API CLIENT
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# 📄 Extract text
def extract_text(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return "".join([p.extract_text() or "" for p in pdf_reader.pages])

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])

        elif uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")

        return None
    except:
        return None

# 📄 PDF
def create_pdf(text):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = []

    for line in text.split("\n"):
        content.append(Paragraph(line, styles["Normal"]))

    doc.build(content)

    with open("report.pdf", "rb") as f:
        return f.read()

# 🎯 UI
st.set_page_config(page_title="HireXpert AI", layout="wide")

st.title("HireXpert AI - Resume Analyzer 🚀")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
job_desc = st.text_area("📄 Paste Job Description (Optional)")

# 🔥 Buttons
col1, col2 = st.columns(2)
analyze = col1.button("Analyze Resume")
improve = col2.button("✨ Improve Resume")

if uploaded_file and (analyze or improve):

    with st.spinner("Processing... ⏳"):

        text = extract_text(uploaded_file)

        if not text:
            st.error("❌ Could not read file")
        else:
            try:
                if analyze:
                    prompt = f"""
                    Analyze this resume based on job description.

                    Job Description:
                    {job_desc}

                    Resume:
                    {text}

                    Give:
                    - Strengths
                    - Weaknesses
                    - ATS Score (out of 100)
                    - Suggestions
                    """

                    response = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=prompt
                    )

                    report = response.text

                    st.subheader("📊 Analysis Result")
                    st.write(report)

                    # score
                    match = re.search(r'(\d{{1,3}})', report)
                    if match:
                        score = int(match.group(1))
                        score = max(0, min(score, 100))
                        st.progress(score / 100)
                        st.write(f"{score}/100")

                    st.download_button("📥 Download TXT", report, "analysis.txt")

                    pdf = create_pdf(report)
                    st.download_button("📄 Download PDF", pdf, "analysis.pdf")

                elif improve:
                    prompt = f"""
                    Rewrite and improve this resume professionally.

                    Job Description:
                    {job_desc}

                    Resume:
                    {text}

                    Make it ATS optimized and well structured.
                    """

                    response = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=prompt
                    )

                    improved = response.text

                    st.subheader("✨ Improved Resume")
                    st.write(improved)

                    st.download_button("📥 Download Improved Resume", improved, "improved_resume.txt")

            except Exception as e:
                st.error(f"Error: {e}")

elif not uploaded_file:
    st.warning("Upload a resume first")

# Footer
st.markdown("---")
st.caption("Made with ❤️ by a student developer")