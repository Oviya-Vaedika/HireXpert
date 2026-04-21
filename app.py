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
    # Keeping '&' for titles like 'IT & Security'
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
            # token_pattern=r"(?u)\b\w+\b" ensures 2-letter words like 'IT' are NOT ignored
            vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform([resume_cleaned, keywords])
            sim = float(cosine_similarity(vec[0:1], vec[1:2])[0][0])
            if sim > highest_sim:
                highest_sim, best_match = sim, industry
        except: continue
    return best_match

# 3. Header
st.markdown("<h1>HireXpert 🌍 <span style='font-size: 0.5em; color: gray;'>Global AI Resume Screening</span></h1>", unsafe_allow_html=True)
st.divider()

# 4. Inputs
jd = st.text_area("Step 1: Paste Job Description (JD):", placeholder="e.g. IT, Sales, or full requirements...", height=150) 
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
                    # token_pattern fixed here too to recognize 'IT'
                    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform([jd_text, resume_text])
                    raw_sim = cosine_similarity(vec[0:1], vec[1:2])
                    score = round(float(raw_sim) * 100, 2)
                    
                    # Logic to prevent 0% for short relevant keywords
                    if any(word in resume_text for word in jd_text.split()) and score < 15:
                        score += 30.0 

                    st.metric("ATS Match Score", f"{min(score, 100.0)}%")
                    st.write(f"**Detected Domain:** {detected_industry}")
                    
                    if score >= 40: st.success("✅ Profile shows relevance.")
                    else: st.warning("⚠️ Match is low. Use the optimization guide on the right.")
                except Exception as e:
                    st.error(f"Analysis error: {e}")

        with col2:
            if st.button("✨ Improve Resume", use_container_width=True):
                st.subheader("JD Optimization Guide")
                
                # --- SYNONYM ENGINE FOR JOB DESCRIPTIONS ---
                jd_synonyms = {
                    "it": ["Information Technology", "Technical Support", "Systems Administration"],
                    "security": ["Cybersecurity", "Network Protection", "Information Assurance"],
                    "sales": ["Business Development", "Account Management", "Revenue Generation"],
                    "hr": ["Human Resources", "Talent Acquisition", "People Operations"],
                    "dev": ["Software Development", "Engineering", "Coding"],
                    "marketing": ["Digital Strategy", "Brand Growth", "Advertising"]
                }
                
                st.write("#### 📝 Keywords to Expand in your Resume:")
                st.write("If the JD mentions short terms, ensure your resume uses these full professional versions:")
                
                found_syns = False
                for word in jd_text.split():
                    if word in jd_synonyms:
                        st.write(f"- For **'{word.upper()}'**, include: *{', '.join(jd_synonyms[word])}*")
                        found_syns = True
                
                if not found_syns:
                    st.info("Tip: Always include both acronyms (like IT) and full titles (Information Technology) to satisfy all ATS filters.")

                st.write("---")
                st.write("#### 🛠 Structural Tips:")
                st.write(f"1. **Align with {detected_industry}:** Ensure your summary mentions your {detected_industry} experience.")
                st.write("2. **Quantify Results:** Don't just say 'Sales', say 'Increased sales by 15%'.")
                st.write("3. **Formatting:** Use a clean, text-based PDF format.")
    else:
        st.error("Could not read the resume text.")
else:
    st.info("Upload your resume and enter a JD to start.")
