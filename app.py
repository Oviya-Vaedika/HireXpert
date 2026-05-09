import streamlit as st
import google.generativeai as genai
import PyPDF2
import docx
import re
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# 🔑 API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ✅ FIXED MODEL
model = genai.GenerativeModel("gemini-1.5-flash-latest")

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


# 🎯 Keyword checker
def keyword_score(text, job_desc):
    base_keywords = [
        "python", "java", "sql", "machine learning",
        "communication", "teamwork", "project",
        "leadership", "analysis", "data"
    ]

    text = text.lower()
    job_desc = job_desc.lower() if job_desc else ""

    keywords = base_keywords + job_desc.split()

    found = [k for k in keywords if k in text]
    score = int((len(found) / len(set(keywords))) * 100)

    return score, list(set(found))


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


# 🎯 UI
st.set_page_config(page_title="HireXpert AI", layout="wide")

st.title("HireXpert AI - Resume Analyzer 🚀")
st.write("Upload your resume and optionally paste a job description")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

# ✅ NEW FEATURE
job_desc = st.text_area("📄 Paste Job Description (Optional)")

# 🚀 Analyze
if uploaded_file and st.button("Analyze Resume"):

    with st.spinner("Analyzing your resume... ⏳"):

        text = extract_text(uploaded_file)

        if not text:
            st.error("❌ Could not read file")
        else:
            try:
                # 🔍 Resume check
                check = model.generate_content(
                    f"Is this a professional resume? Answer ONLY YES or NO:\n{text}"
                )
                result = (check.text or "").upper()

                if result.startswith("NO"):
                    st.error("❌ This is not a resume")
                else:
                    # 🧠 AI Analysis with Job Description
                    response = model.generate_content(
                        f"""
                        Analyze this resume based on the job description.

                        Job Description:
                        {job_desc}

                        Resume:
                        {text}

                        Provide:
                        1. Strengths
                        2. Weaknesses
                        3. ATS Score (out of 100)
                        4. Missing keywords
                        5. Suggestions for improvement
                        """
                    )

                    report = response.text or ""

                    if not report:
                        st.error("⚠️ Failed to generate analysis")
                    else:
                        # 📊 Extract AI score
                        match = re.search(
                            r'(\d{{1,3}})\s*(?:/|out of)?\s*100',
                            report,
                            re.IGNORECASE
                        )

                        ai_score = int(match.group(1)) if match else 50

                        # 🎯 Keyword score
                        k_score, found = keyword_score(text, job_desc)

                        # 🧠 Final score
                        final_score = int((ai_score + k_score) / 2)

                        # 📊 Display score
                        st.subheader("📊 Final ATS Score")
                        st.progress(final_score / 100)
                        st.write(f"{final_score}/100")

                        if job_desc:
                            st.subheader("🎯 Job Match Keywords")
                            st.write(", ".join(found))

                        # 📝 Report
                        st.subheader("Analysis Result")
                        st.write(report)

                        # 📥 TXT download
                        st.download_button(
                            "📥 Download TXT",
                            report,
                            file_name="resume_report.txt"
                        )

                        # 📄 PDF download
                        pdf_data = create_pdf(report)

                        st.download_button(
                            "📄 Download PDF",
                            pdf_data,
                            file_name="resume_report.pdf"
                        )

            except Exception as e:
                st.error(f"Error: {e}")

elif not uploaded_file:
    st.warning("Please upload a resume first.")


#  Footer
st.markdown("---")
st.caption("Made with ❤️ by a student developer")