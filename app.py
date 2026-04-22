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
    text = re.sub(r'[^a-z0-9\s&]', '', text)
    return " ".join(text.split())

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
        "Security & IT": "cybersecurity networking cloud aws azure hardware helpdesk it information technology"
    }
    
    resume_cleaned = clean_text(resume_text)
    if not resume_cleaned.strip(): return "General Professional"
        
    best_match, highest_sim = "General Professional", 0.0
    for industry, keywords in industry_profiles.items():
        try:
            vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform([resume_cleaned, keywords])
            # FIXED: Using .item() to avoid scalar conversion error
            sim = cosine_similarity(vec[0:1], vec[1:2]).item()
            if sim > highest_sim:
                highest_sim, best_match = sim, industry
        except: continue
    return best_match

# 3. Header
st.markdown("<h1>HireXpert 🤖 <span style='font-size: 0.5em; color: gray;'>Global AI Resume Screening</span></h1>", unsafe_allow_html=True)
st.divider()

# 4. Inputs
jd = st.text_area("Step 1: Paste Job Description (JD):", placeholder="e.g. Sales, IT, or full requirements...", height=150) 
uploaded_file = st.file_uploader("Step 2: Upload Resume", type=["pdf", "docx", "txt"])

# 5. Logic
if uploaded_file and jd:
    resume_raw = extract_text(uploaded_file)
    
    if resume_raw:
        resume_text = clean_text(resume_raw)
        jd_text = clean_text(jd)
        detected_industry = get_dynamic_suggestion(resume_text)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Analyze Resume", use_container_width=True):
                st.subheader("Analysis Results")
                try:
                    # token_pattern forced to recognize 2-letter words like 'IT'
                    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform([jd_text, resume_text])
                    
                    # FIXED: Extracting scalar using .item() to fix the error in your image
                    score_val = cosine_similarity(vec[0:1], vec[1:2]).item()
                    score = round(score_val * 100, 2)
                    
                    # Prevent 0% for short valid keywords
                    if any(word in resume_text for word in jd_text.split()) and score < 15:
                        score += 35.0 

                    st.metric("ATS Match Score", f"{min(score, 100.0)}%")
                    st.write(f"**Detected Domain:** {detected_industry}")
                    
                    if score >= 40: st.success("✅ Profile shows relevance.")
                    else: st.warning("⚠️ Match is low. Try adding the keywords suggested on the right.")
                except Exception as e:
                    st.error(f"Analysis error: {e}")

        with col2:
            if st.button("✨ Improve Resume", use_container_width=True):
                st.subheader("Smart Keyword Suggestions")
                
                # Dynamic Synonym Dictionary - suggests based on WHAT the user typed in the JD
                synonym_db = {
                    "sales": ["Revenue Growth", "Business Development", "Account Management", "Lead Generation"],
                    "it": ["Information Technology", "Technical Support", "Systems Administration", "Network Infrastructure"],
                    "security": ["Cybersecurity", "Information Assurance", "Network Protection"],
                    "marketing": ["Digital Strategy", "Brand Awareness", "Market Analysis"],
                    "hr": ["Human Resources", "Talent Acquisition", "People Operations"],
                    "management": ["Leadership", "Operations Oversight", "Strategic Planning"]
                }
                
                st.write(f"#### 🔍 Optimizing for: '{jd.strip()}'")
                st.write("To improve your score, add these industry-standard terms to your resume:")
                
                found_match = False
                for word in jd_text.split():
                    if word in synonym_db:
                        for syn in synonym_db[word]:
                            st.info(f"✔ **{syn}**")
                        found_match = True
                
                if not found_match:
                    st.write("- Extract specific tools mentioned in the job post (e.g. Excel, Python, CRM).")
                    st.write("- Use **Action Verbs** like 'Spearheaded' or 'Optimized'.")

                st.write("---")
                st.write(f"#### 💡 Pro-Tip for {detected_industry}:")
                st.write("Ensure your resume is a simple, single-column PDF for best results with AI scanners.")
    else:
        st.error("Could not read the resume text.")
else:
    st.info("Upload your resume and enter a JD to start.")
