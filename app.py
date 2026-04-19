import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Branding ---
st.set_page_config(page_title="HireXpert", page_icon="🎯", layout="wide")
st.title("HireXpert: Universal AI Resume Scanner 🤖")

# --- Job Library for Suggestions ---
# This acts as the AI's "knowledge" to suggest better fits
JOB_LIBRARY = {
    "Doctor": "medical diagnosis, surgery, patient care, clinical medicine, pharmacology, healthcare",
    "Teacher": "lesson planning, student mentoring, classroom management, education, pedagogy, teaching",
    "Bank Manager": "finance, loan processing, risk management, banking operations, team leadership, accounting",
    "Nurse": "patient monitoring, healthcare, medication, nursing ethics, emergency care, clinical support",
    "Software Engineer": "software development, coding, python, java, technical design, mathematics, cloud",
    "Sales Executive": "marketing, customer relations, negotiation, lead generation, communication, retail"
}

# --- Helper Functions ---
def extract_text(file):
    pdf = PdfReader(file)
    return " ".join([p.extract_text() for p in pdf.pages if p.extract_text()])

def calculate_match(text1, text2):
    tfidf = TfidfVectorizer(stop_words='english')
    vectors = tfidf.fit_transform([text1, text2])
    return round(cosine_similarity(vectors)[0][1] * 100, 2)

# --- UI Layout ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Step 1: Job Requirements")
    # This is where the user pastes any custom job
    job_description = st.text_area("Paste the Job Description here:", height=250, placeholder="Example: Looking for a nurse with ER experience...")

with col2:
    st.subheader("Step 2: Your Profile")
    # This is where they upload the resume
    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

st.divider()

if st.button("Analyze Eligibility & Suggest Careers"):
    if uploaded_resume and job_description.strip():
        with st.spinner("HireXpert AI is analyzing..."):
            resume_text = extract_text(uploaded_resume)
            
            # 1. Check match for the custom pasted job
            main_score = calculate_match(resume_text, job_description)
            
            st.header(f"Job Match Score: {main_score}%")
            
            # 2. Logic for Eligibility
            if main_score >= 50:
                st.success("✅ **ELIGIBLE:** Your resume is a strong match for this specific job!")
            else:
                st.warning("⚠️ **LOW MATCH:** You might not be eligible for this specific role.")
                
                # 3. Suggestion Engine (using the library)
                st.write("---")
                st.subheader("💡 HireXpert Career Suggestions")
                st.write("Based on your resume, our AI found these better fits for you:")
                
                suggestions = []
                for role, desc in JOB_LIBRARY.items():
                    score = calculate_match(resume_text, desc)
                    suggestions.append((role, score))
                
                # Sort by highest match score
                suggestions.sort(key=lambda x: x[1], reverse=True)
                
                # Show top 3 career fits
                for role, score in suggestions[:3]:
                    st.write(f"- **{role}**: {score}% Match")
    else:
        st.error("Please provide both a Job Description and a Resume PDF.")
