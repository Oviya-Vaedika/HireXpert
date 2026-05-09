import streamlit as st
import google.generativeai as genai
import PyPDF2
import docx
import re
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# 🔑 API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# 📄 Extract text
def extract_text(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return "".join([page.extract_text() or "" for page in pdf_reader.pages])

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])

        elif uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")

        return None
    except:
        return None


# 🎯 Basic ATS keyword checker
def keyword_score(text):
    keywords = [
        "python", "java", "sql", "machine learning",
        "communication", "teamwork", "project",
        "leadership", "analysis", "data"
    ]

    text = text.lower()
    found = [k for k in keywords if k in text]
    score = int((len(found) / len(keywords)) * 100)

    return score, found


# 📄 Generate PDF
def create_pdf(report_text):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = []

    for line in report_text.split("\n"):
        content.append(Paragraph(line, styles["Normal"]))

    doc.build(content)

    with open("report.pdf", "rb") as f:
        return f.read()


# UI
st.set_page_config(page_title="HireXpert AI", layout="wide")
st.title("HireXpert AI - Resume Analyzer 🚀")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

if uploaded_file and st.button("Analyze Resume"):

    with st.spinner("Analyzing... ⏳"):

        text = extract_text(uploaded_file)

        if not text:
            st.error("❌ Could not read file")
        else:
            try:
                # Check resume
                check = model.generate_content(
                    f"Is this a resume? Answer YES or NO:\n{text}"
                )
                result = (check.text or "").upper()

                if result.startswith("NO"):
                    st.error("❌ Not a resume")
                else:
                    # AI analysis
                    response = model.generate_content(
                        f"""
                        Analyze this resume:

                        - Strengths
                        - Weaknesses
                        - ATS Score (out of 100)
                        - Suggestions

                        Resume:
                        {text}
                        """
                    )

                    report = response.text or ""

                    # 🔍 Keyword score
                    k_score, found = keyword_score(text)

                    # 📊 Extract AI score
                    match = re.search(r'(\d{1,3})\s*(?:/|out of)?\s*100', report, re.I)

                    ai_score = int(match.group(1)) if match else 50

                    # 🧠 Final score (combined)
                    final_score = int((ai_score + k_score) / 2)

                    # 📊 Display
                    st.subheader("📊 Final ATS Score")
                    st.progress(final_score / 100)
                    st.write(f"{final_score}/100")

                    st.write("✅ Keywords Found:", ", ".join(found))

                    # 📝 Report
                    st.subheader("Analysis")
                    st.write(report)

                    # 📥 TXT download
                    st.download_button(
                        "📥 Download TXT",
                        report,
                        file_name="report.txt"
                    )

                    # 📄 PDF download
                    pdf_data = create_pdf(report)

                    st.download_button(
                        "📄 Download PDF",
                        pdf_data,
                        file_name="report.pdf"
                    )

            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("Made with ❤️ by a student developer")