import streamlit as st
import PyPDF2
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set website branding
st.set_page_config(page_title="HireXpert", page_icon="🌍")
st.title("HireXpert 🌍")
st.markdown("Global AI Resume Screening & Optimization")


def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        return "".join([page.extract_text() or "" for page in pdf_reader.pages])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    return str(uploaded_file.read(), "utf-8")

def get_dynamic_suggestion(resume_text):
    """Analyzes the resume to find the closest matching global industry."""
    # Expanded industry profiles for dynamic matching
    industry_profiles = {
        "Data Science & AI": "python machine learning sql statistics r neural networks analytics",
        "Healthcare & Nursing": "patient care clinical hospital medicine nursing surgery healthcare",
        "Civil & Construction": "structural site management blueprint masonry concrete survey",
        "Finance & Investment": "banking audit portfolio tax accounting excel financial equity",
        "Sales & Business Development": "crm leads negotiation revenue b2b prospecting cold calling",
        "Creative Design": "figma photoshop ui ux adobe illustrator branding creative",
        "Customer Support": "ticketing troubleshooting communication empathy helpdesk service",
        "Logistics & Supply Chain": "warehouse inventory procurement shipping forklift global trade",
        "Hospitality & Culinary": "chef kitchen hotel guest service housekeeping tourism",
        "Law & Legal Services": "litigation contract compliance paralegal judiciary ethics",
        "Digital Marketing": "seo sem social media content strategy copywriting ads google analytics",
        "Human Resources": "recruitment payroll employee relations onboarding talent management",
        "Skilled Trades": "welding electrical plumbing carpentry hvac repair maintenance",
        "Education & Teaching": "lesson plan classroom curriculum pedagogy student teaching",
        "Project Management": "agile scrum jira stakeholder pmp scheduling budget"
    }
    
    best_match = "General Professional"
    highest_sim = 0
    
    # Analyze resume content against all profiles
    for industry, profile_keywords in industry_profiles.items():
        docs = [resume_text.lower(), profile_keywords]
        # Uses TF-IDF to focus on the unique technical keywords
        vec = TfidfVectorizer(stop_words='english').fit_transform(docs)
        sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
        
        if sim > highest_sim:
            highest_sim = sim
            best_match = industry
            
    return best_match

# --- UI Setup ---
st.title("HireXpert 🌍")
jd = st.text_area("Job Description:", height=100)
uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

if uploaded_file and jd:
    resume_text = extract_text(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Analyze Resume"):
            docs = [jd.lower(), resume_text.lower()]
            vec = TfidfVectorizer(stop_words='english').fit_transform(docs)
            score = round(cosine_similarity(vec[0:1], vec[1:2])[0][0] * 100, 2)
            st.metric("Match Score", f"{score}%")
            if score < 50:
                suggestion = get_dynamic_suggestion(resume_text)
                st.error(f"Low match. Based on your skills, try roles in: **{suggestion}**")

    with col2:
        if st.button("✨ Improve Resume"):
            st.subheader("Missing Skills Found in JD:")
            jd_words = set(re.findall(r'\b\w{4,}\b', jd.lower()))
            res_words = set(re.findall(r'\b\w{4,}\b', resume_text.lower()))
            missing = list(jd_words - res_words)
            st.write(", ".join(missing[:12]) if missing else "No missing keywords found!")
