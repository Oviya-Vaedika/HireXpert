import streamlit as st
import PyPDF2
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration
st.set_page_config(page_title="HireXpert", page_icon="🌍", layout="wide")

# 2. Support Functions
def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        return "".join([page.extract_text() or "" for page in pdf_reader.pages])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    return str(uploaded_file.read(), "utf-8")

def get_dynamic_suggestion(resume_text):
    industry_profiles = {
        "Data Science & AI": "python machine learning sql neural networks analytics deep learning",
        "Healthcare & Medicine": "patient clinical medicine nursing surgery hospital healthcare",
        "Construction & Trades": "electrical plumbing masonry carpentry maintenance structural",
        "Finance & Banking": "audit budget tax accounting investment banking portfolio excel",
        "Marketing & SEO": "content strategy ads search engine optimization copywriting marketing",
        "Sales & Business": "crm leads negotiation revenue b2b prospecting cold calling",
        "Design & UX/UI": "figma photoshop adobe illustrator branding creative prototype",
        "Supply Chain & Logistics": "warehouse inventory shipping procurement forklift logistics",
        "Law & Legal": "litigation contract compliance paralegal judiciary ethics",
        "Education & Teaching": "lesson plan curriculum classroom pedagogy student teaching",
        "Human Resources": "recruitment payroll onboarding talent management employee relations",
        "Customer Service": "ticketing communication empathy helpdesk troubleshooting",
        "Hospitality": "chef kitchen hotel guest service housekeeping tourism culinary",
        "Project Management": "agile scrum jira pmp stakeholder scheduling budget",
        "Security & IT": "cybersecurity networking cloud aws azure hardware helpdesk"
    }
    best_match, highest_sim = "General Professional", 0
    for industry, keywords in industry_profiles.items():
        docs = [resume_text.lower(), keywords]
        vec = TfidfVectorizer(stop_words='english').fit_transform(docs)
        sim = cosine_similarity(vec[0:1], vec[1:2])
        if sim > highest_sim:
            highest_sim, best_match = sim, industry
    return best_match

# 3. Header with Subtitle
# Using columns to place the subtitle next to the main title
head1, head2 = st.columns([0.3, 0.7])
with head1:
    st.title("HireXpert 🌍")
with head2:
    st.markdown("<br><h4 style='color: grey;'>Global AI Resume Screening & Optimization</h4>", unsafe_content_allowed=True)

st.divider()

# 4. Inputs (Larger JD Box)
jd = st.text_area("Step 1: Paste the Job Description (JD):", 
                  placeholder="Paste requirements for ANY job in the world here...", 
                  height=300) # Increased height

uploaded_file = st.file_uploader("Step 2: Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# 5. Logic
if uploaded_file and jd:
    resume_text = extract_text(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Analyze Resume", use_container_width=True):
            st.subheader("Analysis Results")
            docs = [jd.lower(), resume_text.lower()]
            vec = TfidfVectorizer(stop_words='english').fit_transform(docs)
            score = round(cosine_similarity(vec[0:1], vec[1:2])[0][0] * 100, 2)
            
            st.metric("ATS Match Score", f"{score}%")
            if score >= 60:
                st.success("✅ Eligible: Your profile is a strong match.")
            else:
                industry = get_dynamic_suggestion(resume_text)
                st.error(f"❌ Low Match. Suggested Field: **{industry}**")

    with col2:
        if st.button("✨ Improve Resume", use_container_width=True):
            st.subheader("Optimization Guide")
            jd_words = set(re.findall(r'\b\w{5,}\b', jd.lower()))
            res_words = set(re.findall(r'\b\w{5,}\b', resume_text.lower()))
            missing = list(jd_words - res_words)
            
            st.write("#### 🛠 Keywords to Add:")
            st.info(", ".join(missing[:15]) if missing else "No major keywords missing!")
            
            st.write("#### 💡 Quick Tips:")
            st.write("- Use **bold** for key skills.\n- Include specific **metrics** (e.g., 'saved 20% time').\n- Avoid using complex **graphics**.")
else:
    st.info("Fill in the Job Description and upload a file to begin.")
