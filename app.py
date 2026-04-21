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

def get_dynamic_suggestion(resume_text):
    """Matches resume text against 15 industry profiles to suggest a career path."""
    industry_profiles = {
        "Data Science & AI": "python machine learning sql neural networks analytics deep learning data visualization pytorch tensorflow",
        "Healthcare & Medicine": "patient clinical medicine nursing surgery hospital healthcare medical records diagnostics",
        "Construction & Trades": "electrical plumbing masonry carpentry maintenance structural blueprint safety osha",
        "Finance & Banking": "audit budget tax accounting investment banking portfolio excel sap fintech reporting",
        "Marketing & SEO": "content strategy ads search engine optimization copywriting marketing google analytics social media campaigns",
        "Sales & Business": "crm leads negotiation revenue b2b prospecting cold calling salesforce account management",
        "Design & UX/UI": "figma photoshop adobe illustrator branding creative prototype user experience wireframing",
        "Supply Chain & Logistics": "warehouse inventory shipping procurement forklift logistics distribution operations",
        "Law & Legal": "litigation contract compliance paralegal judiciary ethics research documentation",
        "Education & Teaching": "lesson plan curriculum classroom pedagogy student teaching tutoring academic",
        "Human Resources": "recruitment payroll onboarding talent management employee relations performance review",
        "Customer Service": "ticketing communication empathy helpdesk troubleshooting client support",
        "Hospitality": "chef kitchen hotel guest service housekeeping tourism culinary restaurant management",
        "Project Management": "agile scrum jira pmp stakeholder scheduling budget planning leadership kanban",
        "Security & IT": "cybersecurity networking cloud aws azure hardware helpdesk firewalls infrastructure"
    }
    
    if not resume_text.strip():
        return "General Professional"
        
    best_match, highest_sim = "General Professional", 0.0
    
    for industry, keywords in industry_profiles.items():
        try:
            docs = [resume_text.lower(), keywords.lower()]
            vec = TfidfVectorizer(stop_words='english').fit_transform(docs)
            sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
            if sim > highest_sim:
                highest_sim, best_match = sim, industry
        except:
            continue
    return best_match

# 3. Custom Header
st.markdown("""
    <div style='display: flex; align-items: baseline;'>
        <h1 style='margin-right: 15px;'>HireXpert 🌍</h1>
        <h4 style='color: gray; font-weight: normal; opacity: 0.8;'>Global AI Resume Screening & Optimization</h4>
    </div>
""", unsafe_allow_html=True)
st.divider()

# 4. Input Section
jd = st.text_area("Step 1: Paste the Job Description (JD):", 
                  placeholder="Paste requirements for ANY job in the world here...", 
                  height=250) 

uploaded_file = st.file_uploader("Step 2: Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# 5. Logic & Output
if uploaded_file and jd:
    resume_text = extract_text(uploaded_file)
    
    if resume_text:
        col1, col2 = st.columns(2)
        
        # Determine industry early for use in both columns
        detected_industry = get_dynamic_suggestion(resume_text)
        
        with col1:
            if st.button("🔍 Analyze Resume", use_container_width=True):
                st.subheader("Analysis Results")
                try:
                    docs = [jd.lower(), resume_text.lower()]
                    vec = TfidfVectorizer(stop_words='english').fit_transform(docs)
                    raw_score = cosine_similarity(vec[0:1], vec[1:2])[0][0]
                    score = round(float(raw_score) * 100, 2)
                    
                    st.metric("ATS Match Score", f"{score}%")
                    st.write(f"**Detected Domain:** {detected_industry}")
                    
                    if score >= 60:
                        st.success("✅ Eligible: Your profile is a strong match for this JD.")
                    else:
                        st.warning(f"⚠️ Low Match. Your background is heavily weighted toward: **{detected_industry}**")
                except:
                    st.error("Analysis failed. Ensure both the JD and Resume contain descriptive text.")

        with col2:
            if st.button("✨ Improve Resume", use_container_width=True):
                st.subheader("Optimization Guide")
                
                # Dynamic advice based on detected industry
                st.write(f"#### 🎯 Profile Insight: **{detected_industry}**")
                
                industry_tips = {
                    "Data Science & AI": "Showcase projects using specific frameworks like PyTorch or Scikit-learn. Highlight business impact (e.g., 'Improved accuracy by 10%').",
                    "Healthcare & Medicine": "Ensure clinical certifications and specialized software (EHR/EMR) are listed in your skills section.",
                    "Construction & Trades": "Focus on safety records, specific equipment licenses, and project completion timelines.",
                    "Finance & Banking": "Detail your experience with regulatory compliance and proficiency in advanced financial modeling.",
                    "Marketing & SEO": "Use hard data. Mention growth percentages, CPC, and specific tools like Hubspot or SEMrush.",
                    "Sales & Business": "Quantify your achievements. Focus on quota attainment, revenue growth, and CRM management.",
                    "Design & UX/UI": "Ensure your portfolio link is active. Emphasize user research and iterative design processes.",
                    "Project Management": "Highlight leadership in cross-functional teams and mastery of Agile or Scrum methodologies.",
                    "Security & IT": "List your technical certifications (CompTIA, CISSP, AWS) and specific network architecture experience.",
                    "Human Resources": "Emphasize experience with ATS platforms, employee retention strategies, and conflict resolution."
                }

                # Get specific tip or default to a general one
                tip = industry_tips.get(detected_industry, "Focus on quantifiable achievements (e.g., 'Reduced costs by 15%') and industry-standard certifications.")
                
                st.info(f"**Actionable Tip:** {tip}")
                
                st.write("#### 🛠 Structural Improvements:")
                st.write("- **Quantify:** Replace vague tasks with 'Improved [X] by [Y]% by doing [Z]'.")
                st.write("- **Readability:** Use bullet points. Avoid paragraphs and complex multi-column layouts.")
                st.write("- **Keywords:** Ensure the specific tools mentioned in the Job Description appear in your skills section.")
    else:
        st.error("Could not extract text from the file.")
else:
    st.info("Please provide both a Job Description and a Resume to begin.")
