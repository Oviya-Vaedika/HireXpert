import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# --- Page Branding ---
st.set_page_config(page_title="HireXpert AI", page_icon="🎯", layout="wide")
st.title("HireXpert: Advanced ATS Resume Scanner 🤖")

# --- Expanded Job Library ---
JOB_LIBRARY = {
    "Doctor": "medical diagnosis, surgery, patient care, clinical medicine, pharmacology, healthcare, physician",
    "Teacher": "lesson planning, student mentoring, classroom management, education, pedagogy, teaching, school",
    "Bank Manager": "finance, loan processing, risk management, banking operations, team leadership, accounting, credit",
    "Nurse": "patient monitoring, healthcare, medication, nursing ethics, emergency care, clinical support, hospital",
    "Software Engineer": "software development, coding, python, java, technical design, mathematics, cloud, developer",
    "Sales Executive": "marketing, customer relations, negotiation, lead generation, communication, retail, targets",
    "Data Analyst": "excel, sql, statistics, data visualization, powerbi, reporting, python, analytics",
    "Police Officer": "law enforcement, public safety, patrol, investigation, emergency response, criminal justice",
    "Chef": "culinary arts, food safety, menu planning, kitchen management, cooking, gastronomy"
}

# --- Helper Functions ---
def extract_text(file):
    file_extension = file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        pdf = PdfReader(file)
        return " ".join([p.extract_text() for p in pdf.pages if p.extract_text()])
    elif file_extension in ['doc', 'docx']:
        doc = Document(io.BytesIO(file.read()))
        return " ".join([para.text for para in doc.paragraphs])
    return ""

def is_resume(text):
    # Professional validation: looks for common resume headers
    resume_keywords = ["experience", "education", "skills", "summary", "contact", "projects", "work"]
    count = sum(1 for word in resume_keywords if word in text.lower())
    return count >= 2  # Must have at least 2 common resume sections

def calculate_match(text1, text2):
    tfidf = TfidfVectorizer(stop_words='english')
    vectors = tfidf.fit_transform([text1, text2])
    return round(cosine_similarity(vectors)[0][1] * 100, 2)

# --- UI Layout ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Job Requirements")
    job_description = st.text_area("Paste the Job Description here:", height=250, placeholder="Paste requirements here...")

with col2:
    st.subheader("📄 Upload Profile")
    uploaded_file = st.file_uploader("Upload Resume (PDF or Word)", type=["pdf", "docx", "doc"])

st.divider()

if st.button("Analyze ATS Score & Eligibility"):
    if uploaded_file and job_description.strip():
        with st.spinner("HireXpert is calculating your ATS Score..."):
            resume_text = extract_text(uploaded_file)
            
            # Check if it's actually a resume
            if not is_resume(resume_text):
                st.error("❌ Error: The uploaded file does not appear to be a Resume. Please ensure it includes sections like 'Experience' or 'Education'.")
            else:
                # Calculate ATS Score
                ats_score = calculate_match(resume_text, job_description)
                
                st.header(f"ATS Match Score: {ats_score}%")
                
                if ats_score >= 75:
                    st.success("🌟 **EXCELLENT MATCH!** Your resume is perfectly optimized for this role.")
                elif ats_score >= 50:
                    st.info("✅ **GOOD ELIGIBILITY:** You have a solid foundation. Consider adding a few more specific keywords.")
                else:
                    st.warning("⚠️ **LOW MATCH:** This specific role might be a challenge, but don't give up! Every expert was once a beginner. Your unique journey will lead you to the right door! ✨")
                    
                    # Suggestion Engine
                    st.write("---")
                    st.subheader("🔍 HireXpert Smart Career Suggestions")
                    st.write("Don't be discouraged! Our AI found these roles match your current skills better:")
                    
                    suggestions = []
                    for role, desc in JOB_LIBRARY.items():
                        score = calculate_match(resume_text, desc)
                        suggestions.append((role, score))
                    
                    suggestions.sort(key=lambda x: x[1], reverse=True)
                    for role, score in suggestions[:3]:
                        st.write(f"- **{role}**: {score}% Match")
    else:
        st.error("Please provide both a Job Description and a Resume file.")
