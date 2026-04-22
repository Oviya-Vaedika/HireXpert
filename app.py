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
        
        # VALIDATION
        resume_indicators = ['experience', 'education', 'skills', 'projects', 'summary', 'contact']
        is_valid = any(indicator in resume_text_clean for indicator in resume_indicators)

        if not is_valid:
            st.error("❌ The file uploaded does not seem to be a standard Resume.")
        else:
            detected_industry, industry_keywords = get_dynamic_suggestion(resume_text_clean)
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Analyze Resume", use_container_width=True):
                    st.subheader("Analysis Results")
                    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform([jd_text, resume_text_clean])
                    score = round(cosine_similarity(vec[0:1], vec[1:2]).item() * 100, 2)
                    if any(word in resume_text_clean for word in jd_text.split()) and score < 15: score += 35.0 
                    st.metric("ATS Match Score", f"{min(score, 100.0)}%")
                    st.write(f"**Detected Domain:** {detected_industry}")

            with col2:
                if st.button("✨ Improve Resume", use_container_width=True):
                    st.subheader(f"Gap Analysis for your {detected_industry} Profile")
                    
                    # --- DYNAMIC COMPONENT 1: MISSING KEYWORDS ---
                    res_words = set(resume_text_clean.split())
                    ind_words = set(industry_keywords.split())
                    missing = list(ind_words - res_words)
                    
                    if missing:
                        st.write("#### 🛠 Missing Industry Keywords:")
                        st.write("Your resume is missing these core terms for your field. Adding them will boost your score:")
                        st.info(", ".join(missing[:6]))
                    
                    # --- DYNAMIC COMPONENT 2: WORD REPLACEMENT ---
                    weak_words = {
                        "managed": "Spearheaded",
                        "helped": "Facilitated",
                        "led": "Orchestrated",
                        "responsible": "Accountable",
                        "worked": "Executed"
                    }
                    
                    st.write("#### 📝 Personalized Word Swaps:")
                    found_weak = False
                    for weak, strong in weak_words.items():
                        if weak in resume_text_clean:
                            st.write(f"- You used **'{weak}'**. Replace it with: **'{strong}'**.")
                            found_weak = True
                    
                    if not found_weak:
                        st.success("Great job! Your resume already uses strong action verbs.")

                    # --- DYNAMIC COMPONENT 3: SPECIFIC ADVICE ---
                    st.write("---")
                    if "experience" in resume_text_clean and len(resume_raw) < 1500:
                        st.warning("💡 **Suggestion:** Your resume is a bit short. Add more quantifiable achievements (e.g., numbers, % or $).")
                    else:
                        st.info("💡 **Suggestion:** Your resume length is good. Ensure your most recent role has at least 4-5 bullet points.")
    else:
        st.error("Could not extract text.")
else:
    st.info("Upload your resume and enter a JD to start.")
