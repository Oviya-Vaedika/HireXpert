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
    """Safely extracts text from PDF, DOCX, or TXT files."""
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
    """Basic cleaning to improve matching accuracy."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text

def get_dynamic_suggestion(resume_text):
    """Matches resume text against industry profiles."""
    industry_profiles = {
        "Data Science & AI": "python machine learning sql analytics deep learning data visualization pytorch tensorflow",
        "Healthcare & Medicine": "patient clinical medicine nursing surgery hospital healthcare medical records",
        "Finance & Banking": "audit budget tax accounting investment banking portfolio excel sap fintech",
        "Marketing & SEO": "content strategy ads search engine optimization copywriting marketing analytics social media",
        "Sales & Business": "crm leads negotiation revenue b2b prospecting salesforce account management",
        "Design & UX/UI": "figma photoshop adobe illustrator branding creative prototype user experience wireframing",
        "Project Management": "agile scrum jira pmp stakeholder scheduling budget planning leadership",
        "Security & IT": "cybersecurity networking cloud aws azure hardware helpdesk firewalls infrastructure it security"
    }
    
    resume_cleaned = clean_text(resume_text)
    if not resume_cleaned.strip():
        return "General Professional"
        
    best_match, highest_sim = "General Professional", 0.0
    
    for industry, keywords in industry_profiles.items():
        try:
            # We don't use stop_words here to ensure "IT" is caught
            vec = TfidfVectorizer(ngram_range=(1, 2)).fit_transform([resume_cleaned, keywords])
            sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
            if sim > highest_sim:
                highest_sim, best_match = sim, industry
        except:
            continue
    return best_match

# 3. Header
st.markdown("""
    <div style='display: flex; align-items: baseline;'>
        <h1 style='margin-right: 15px;'>HireXpert 🌍</h1>
        <h4 style='color: gray; font-weight: normal; opacity: 0.8;'>Global AI Resume Screening & Optimization</h4>
    </div>
""", unsafe_allow_html=True)
st.divider()

# 4. Inputs
jd = st.text_area("Step 1: Paste the Job Description (JD):", 
                  placeholder="Note: Short JDs like 'IT & Security' may yield lower scores than full descriptions.", 
                  height=200) 

uploaded_file = st.file_uploader("Step 2: Upload Resume", type=["pdf", "docx", "txt"])

# 5. Logic
if uploaded_file and jd:
    resume_raw = extract_text(uploaded_file)
    
    if resume_raw:
        resume_text = clean_text(resume_raw)
        jd_text = clean_text(jd)
        
        col1, col2 = st.columns(2)
        detected_industry = get_dynamic_suggestion(resume_text)
        
        with col1:
            if st.button("🔍 Analyze Resume", use_container_width=True):
                st.subheader("Analysis Results")
                try:
                    # ngram_range=(1,2) helps catch phrases like "IT Security"
                    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
                    vec = vectorizer.fit_transform([jd_text, resume_text])
                    raw_score = cosine_similarity(vec[0:1], vec[1:2])[0][0]
                    
                    # Boost score slightly if keywords match to avoid 0% on short JDs
                    score = round(float(raw_score) * 100, 2)
                    
                    st.metric("ATS Match Score", f"{score}%")
                    st.write(f"**Detected Domain:** {detected_industry}")
                    
                    if score >= 50:
                        st.success("✅ Eligible: Good match.")
                    else:
                        st.warning(f"⚠️ Low Match for this specific JD. Your profile fits better in: **{detected_industry}**")
                except:
                    st.error("Analysis failed. Try a longer Job Description.")

        with col2:
            if st.button("✨ Improve Resume", use_container_width=True):
                st.subheader("Optimization Guide")
                st.write(f"#### 🎯 Profile Insight: **{detected_industry}**")
                
                tips = {
                    "Security & IT": "Focus on certifications like CompTIA, CISSP, or AWS. Explicitly list firewall and network protocols.",
                    "Data Science & AI": "Showcase Python projects and specific ML models you've deployed.",
                    "Project Management": "Highlight Agile/Scrum certifications and team sizes you've managed."
                }
                
                st.info(tips.get(detected_industry, "Focus on adding measurable achievements and industry-standard keywords."))
                st.write("---")
                st.write("**Quick Tip:** Short Job Descriptions (like just a title) make it hard for the AI to find matches. Try pasting the full 'Requirements' section of the job post!")
    else:
        st.error("Text extraction failed.")
else:
    st.info("Please upload a resume and paste a job description.")
