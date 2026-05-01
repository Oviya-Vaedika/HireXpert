import streamlit as st
import google.generativeai as genai
import PyPDF2
import docx
import re

# 🔑 Load Gemini API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# 🤖 Load model
model = genai.GenerativeModel("gemini-1.5-flash")

# 📄 Function to extract text
def extract_text(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return "".join([page.extract_text() or "" for page in pdf_reader.pages])

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])

        elif uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")

        else:
            return None

    except Exception as e:
        st.error(f"File reading error: {e}")
        return None


# 🎯 UI
st.title("HireXpert AI - Resume Analyzer")
st.write("Upload your resume and paste job description for better ATS analysis")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

# 🧾 NEW: Job Description box
job_description = st.text_area("Paste Job Description Here (optional)", height=200)

# 🚀 Analyze Button
if st.button("Analyze Resume"):

    if uploaded_file is None:
        st.warning("Please upload a resume first.")
    else:
        resume_text = extract_text(uploaded_file)

        if not resume_text:
            st.error("❌ Unsupported or unreadable file.")
        else:
            try:
                # 🧠 Check if resume
                check_prompt = f"""
                Is the following text a professional resume?
                Answer ONLY YES or NO.

                Text:
                {resume_text}
                """
                check_response = model.generate_content(check_prompt)
                result = check_response.text.strip().upper()

                if "NO" in result:
                    st.error("❌ This is not a resume.")
                else:

                    # 🧠 ATS Analysis with Job Description
                    analysis_prompt = f"""
                    You are an ATS resume expert.

                    Compare the resume with the job description and provide:

                    1. Match Score (0-100)
                    2. Strengths
                    3. Weaknesses
                    4. Missing Keywords
                    5. Suggestions to improve ATS ranking

                    Resume:
                    {resume_text}

                    Job Description:
                    {job_description if job_description else "Not provided"}
                    """

                    response = model.generate_content(analysis_prompt)
                    report = response.text

                    # 📊 Extract score
                    match = re.search(r'(\d{1,3})\s*/?\s*100', report)

                    if match:
                        score = int(match.group(1))
                        score = max(0, min(score, 100))

                        st.subheader("📊 ATS Match Score")
                        st.progress(score / 100)
                        st.write(f"{score}/100")

                    # 📝 Result
                    st.subheader("Analysis Result")
                    st.write(report)

                    # 📥 Download
                    st.download_button(
                        "📥 Download Report",
                        data=report,
                        file_name="resume_analysis.txt"
                    )

            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("Made with ❤️ by a student developer")
