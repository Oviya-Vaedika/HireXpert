import streamlit as st
import PyPDF2
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration
st.set_page_config(page_title="HireXpert", page_icon="🤖", layout="wide")

# 2. Support Functions
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
    expansions = {
        r'\bit\b': 'information technology', r'\bjd\b': 'job description',
        r'\bai\b': 'artificial intelligence', r'\bml\b': 'machine learning',
        r'\bqa\b': 'quality assurance', r'\bui\b': 'user interface',
        r'\bux\b': 'user experience', r'\bhr\b': 'human resources',
        r'\bsw\b': 'software', r'\bhw\b': 'hardware'
    }
    for abbrev, full in expansions.items():
        text = re.sub(abbrev, full, text)
    text = re.sub(r'[^a-z0-9\s&+#.]', '', text)
    return " ".join(text.split())

def get_dynamic_suggestion(resume_text):
    industry_profiles = {
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
    resume_cleaned = clean_text(resume_text)
    if not resume_cleaned.strip(): return "General Professional", ""
    best_match, highest_sim, matching_keywords = "General Professional", 0.0, ""
    for industry, keywords in industry_profiles.items():
        try:
            vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform([resume_cleaned, keywords])
            sim = cosine_similarity(vec[0:1], vec[1:2]).item()
            if sim > highest_sim:
                highest_sim, best_match, matching_keywords = sim, industry, keywords
        except: continue
    return best_match, matching_keywords

# 3. Custom Header
st.markdown("<h1>HireXpert 🤖 <span style='font-size: 0.5em; color: gray;'>Global AI Resume Screening</span></h1>", unsafe_allow_html=True)
st.divider()

# 4. Input Section
jd = st.text_area("Step 1: Paste the Job Description (JD):", placeholder="e.g. Sales, IT, or full requirements...", height=150) 
uploaded_file = st.file_uploader("Step 2: Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# 5. Logic & Output
if uploaded_file and jd:
    resume_raw = extract_text(uploaded_file)
    if resume_raw:
        resume_text_clean = clean_text(resume_raw)
        jd_text = clean_text(jd)
        resume_indicators = ['experience', 'education', 'skills', 'projects', 'summary', 'contact']
        is_valid = len([word for word in resume_indicators if word in resume_text_clean]) >= 3 

        if not is_valid:
            st.error("❌ The file uploaded does not seem to be a Resume.")
        else:
            detected_industry, industry_keywords = get_dynamic_suggestion(resume_text_clean)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Analyze Resume", use_container_width=True):
                    st.subheader("Analysis Results")
                    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform([jd_text, resume_text_clean])
                    score = round(cosine_similarity(vec[0:1], vec[1:2]).item() * 100, 2)
                    st.metric("ATS Match Score", f"{min(score, 100.0)}%")
                    st.write(f"**Detected Domain:** {detected_industry}")
            with col2:
                if st.button("✨ Improve Resume", use_container_width=True):
                    st.subheader(f"Gap Analysis")
                    res_words = set(resume_text_clean.split())
                    ind_words = set(industry_keywords.split())
                    missing = list(ind_words - res_words)
                    if missing:
                        st.write("#### 🛠 Missing Industry Keywords:")
                        st.info(", ".join(missing[:6]))
                    st.write("#### 📝 Word Swaps:")
                    weak_words = {"managed": "Spearheaded", "helped": "Facilitated", "led": "Orchestrated"}
                    for weak, strong in weak_words.items():
                        if weak in resume_text_clean:
                            st.write(f"- Replace **'{weak}'** with: **'{strong}'**.")

            # Defined a placeholder for report so the code runs
            report = "Resume Analysis Report Content" 
            st.download_button(
                "📥 Download Report",
                report,
                file_name="hirexpert_report.txt"
            )
    else:
        st.error("Could not extract text.")
else:
    st.info("Upload your resume and enter a JD to start.")

# ---------------- FOOTER ----------------
st.divider()
st.caption("Made with ❤️ by a student developer")
