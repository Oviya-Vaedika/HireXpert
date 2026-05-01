import streamlit as st
import PyPDF2
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="HireXpert AI", page_icon="🤖", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
.sub-text {
    text-align: center;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# ---------------- FUNCTIONS ----------------
def extract_text(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return "".join([page.extract_text() or "" for page in pdf_reader.pages])
        elif uploaded_file.type.endswith("document"):
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            return str(uploaded_file.read(), "utf-8")
    except:
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return " ".join(text.split())

def calculate_score(jd, resume):
    vec = TfidfVectorizer().fit_transform([jd, resume])
    return round(cosine_similarity(vec[0:1], vec[1:2])[0][0] * 100, 2)

def get_missing_keywords(jd, resume):
    jd_words = set(jd.split())
    res_words = set(resume.split())
    return list(jd_words - res_words)

def detect_industry(resume):
    industries = {
        "Data Science & AI": "python machine learning sql neural networks analytics deep learning pytorch tensorflow",
        "Healthcare & Medicine": "patient clinical medicine nursing surgery hospital healthcare medical",
        "Construction & Trades": "electrical plumbing masonry carpentry maintenance structural safety osha",
        "Finance & Banking": "audit budget tax accounting investment banking portfolio excel fintech",
        "Marketing & SEO": "content strategy ads search engine optimization copywriting marketing analytics",
        "Sales & Business": "crm leads negotiation revenue b2b prospecting cold calling salesforce",
        "Design & UX/UI": "figma photoshop adobe illustrator branding creative prototype wireframing",
        "Project Management": "agile scrum jira pmp stakeholder scheduling budget kanban",
        "Security & IT": "cybersecurity networking cloud aws azure hardware helpdesk it infrastructure"
    }

    best_match = "General"
    highest_score = 0

    for industry, keywords in industries.items():
        vec = TfidfVectorizer().fit_transform([resume, keywords])
        score = cosine_similarity(vec[0:1], vec[1:2])[0][0]
        if score > highest_score:
            highest_score = score
            best_match = industry

    return best_match

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🚀 HireXpert AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Smart Resume Analyzer for Students</div>", unsafe_allow_html=True)

st.divider()

# ---------------- INPUT ----------------
col1, col2 = st.columns(2)

with col1:
    jd = st.text_area("📄 Paste Job Description", height=200)

with col2:
    uploaded_file = st.file_uploader("📎 Upload Resume", type=["pdf", "docx", "txt"])

# ---------------- MAIN BUTTON ----------------
if st.button("🔍 Analyze Resume", use_container_width=True):

    if not jd or not uploaded_file:
        st.warning("⚠ Please provide both inputs")
    else:
        resume_raw = extract_text(uploaded_file)

        if not resume_raw:
            st.error("❌ Could not read file")
        else:
            jd_clean = clean_text(jd)
            resume_clean = clean_text(resume_raw)

            score = calculate_score(jd_clean, resume_clean)
            industry = detect_industry(resume_clean)
            missing_keywords = get_missing_keywords(jd_clean, resume_clean)

            # ---------------- RESULTS ----------------
            st.subheader("📊 Analysis Results")

            colA, colB = st.columns(2)

            with colA:
                st.metric("ATS Score", f"{score}%")
                st.progress(int(score))

            with colB:
                st.info(f"🧠 Detected Industry: {industry}")

            # ---------------- SCORE FEEDBACK ----------------
            if score > 75:
                st.success("🔥 Excellent! Your resume is strong.")
            elif score > 50:
                st.warning("⚡ Decent, but needs improvement.")
            else:
                st.error("❌ Low match. Improve your resume.")

            st.divider()

            # ---------------- KEYWORD GAP ----------------
            st.subheader("🛠 Missing Keywords")

            if missing_keywords:
                st.info(", ".join(missing_keywords[:10]))
            else:
                st.success("✅ No major keywords missing!")

            st.divider()

            # ---------------- SUGGESTIONS ----------------
            st.subheader("✨ Resume Tips")

            tips = [
                "Use strong action verbs (e.g., Built, Developed, Led)",
                "Add measurable results (e.g., Increased efficiency by 30%)",
                "Match keywords with job description",
                "Keep formatting simple for ATS",
                "Avoid long paragraphs"
            ]

            for tip in tips:
                st.write(f"- {tip}")

            st.divider()

            # ---------------- DOWNLOAD REPORT ----------------
            report = f"""
HireXpert AI Report
Date: {datetime.now()}

ATS Score: {score}%
Detected Industry: {industry}

Missing Keywords:
{', '.join(missing_keywords[:10])}

Tips:
- Use strong action verbs
- Add measurable achievements
- Match job keywords
"""

            st.download_button(
                "📥 Download Report",
                report,
                file_name="hirexpert_report.txt"
            )

# ---------------- FOOTER ----------------
st.divider()
st.caption("Made with ❤️ by a student developer")