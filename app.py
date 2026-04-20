import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# --- 1. Page Config & Professional Styling ---
st.set_page_config(page_title="HireXpert AI", page_icon="🎯", layout="wide")
st.title("HireXpert: Professional ATS Optimizer 🤖")

# --- 2. Professional Global Job Library ---
# Expanded for career suggestions
JOB_LIBRARY = {
    "Doctor": "medical diagnosis surgery patient care medicine clinical pharmacology healthcare physician hospital",
    "Teacher": "lesson planning student mentoring classroom management education pedagogy teaching school curriculum",
    "Bank Manager": "finance loan processing risk management banking operations team leadership accounting credit audit",
    "Nurse": "patient monitoring healthcare medication nursing ethics emergency care clinical support hospital nursing",
    "Software Engineer": "software development coding python java technical design mathematics cloud developer github",
    "Data Analyst": "excel sql statistics data visualization powerbi reporting python analytics dashboards",
    "Police Officer": "law enforcement public safety patrol investigation emergency response criminal justice security",
    "Chef": "culinary arts food safety menu planning kitchen management cooking gastronomy restaurant",
    "Sales Executive": "marketing customer relations negotiation lead generation communication retail targets b2b"
}

# --- 3. Advanced Helper Functions ---

def extract_text(file):
    """Reads both PDF and Word files correctly."""
    file_extension = file.name.split('.')[-1].lower()
    text = ""
    try:
        if file_extension == 'pdf':
            pdf = PdfReader(file)
            text = " ".join([p.extract_text() for p in pdf.pages if p.extract_text()])
        elif file_extension in ['doc', 'docx']:
            doc = Document(io.BytesIO(file.read()))
            text = " ".join([para.text for para in doc.paragraphs])
    except Exception:
        return ""
    return text.strip()

def analyze_formatting(text):
    """Calculates if the resume is built properly for machines."""
    # Standard headers that ATS systems look for
    standard_headers = ["experience", "education", "skills", "projects", "summary", "contact", "certifications"]
    found_headers = [h for h in standard_headers if h in text.lower()]
    
    # Points awarded for each standard section found
    formatting_score = (len(found_headers) / len(standard_headers)) * 100
    return round(formatting_score, 0), found_headers

def calculate_match(text1, text2):
    """Calculates the keyword similarity score."""
    if not text1 or not text2: return 0.0
    tfidf = TfidfVectorizer(stop_words='english')
    vectors = tfidf.fit_transform([text1.lower(), text2.lower()])
    return round(cosine_similarity(vectors)[0][1] * 100, 2)

# --- 4. Main Website Interface ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Targeted Job Description")
    job_desc = st.text_area("Paste the job requirements from LinkedIn or Indeed:", height=250)

with col2:
    st.subheader("📄 Your Professional Resume")
    uploaded_file = st.file_uploader("Upload Profile (PDF or Word)", type=["pdf", "docx", "doc"])

st.divider()

if st.button("Generate HireXpert ATS Report"):
    if uploaded_resume_file := uploaded_file:
        if job_desc.strip():
            with st.spinner("Analyzing profile structure and keyword alignment..."):
                raw_text = extract_text(uploaded_resume_file)
                
                # Check if it's a real resume (basic validation)
                if len(raw_text) < 100 or "education" not in raw_text.lower() and "experience" not in raw_text.lower():
                    st.error("❌ Invalid File: This does not appear to be a professional resume. Please ensure it includes 'Education' or 'Experience' sections.")
                else:
                    # 1. Formatting Check
                    format_score, sections = analyze_formatting(raw_text)
                    
                    # 2. Keyword Match Check
                    match_score = calculate_match(raw_text, job_desc)
                    
                    # 3. Final Weighted ATS Score
                    # 60% Keyword matching + 40% Formatting quality
                    final_ats_score = round((match_score * 0.6) + (format_score * 0.4), 2)
                    
                    # --- DISPLAY RESULTS ---
                    st.header(f"Final ATS Score: {final_ats_score}%")
                    
                    res_col1, res_col2 = st.columns(2)
                    res_col1.metric("Keyword Match", f"{match_score}%")
                    res_col2.metric("Formatting Score", f"{format_score}%")
                    
                    st.write(f"**Sections Detected:** {', '.join(sections).title()}")

                    if final_ats_score >= 70:
                        st.success("🌟 **EXCELLENT:** Your resume is highly optimized and ready for submission!")
                    elif final_ats_score >= 40:
                        st.info("✅ **DECENT:** You have a good foundation, but try adding more specific keywords from the job description.")
                    else:
                        st.warning("⚠️ **NOT QUITE ELIGIBLE:** This specific role might be a tough fit right now, but don't lose heart! Every rejection is just redirection to the right path. Your perfect opportunity is still waiting! ✨")
                        
                        # Career Suggestion Engine
                        st.divider()
                        st.subheader("💡 HireXpert Smart Career Alternatives")
                        st.write("Based on your existing skills, you are a much stronger fit for these roles:")
                        
                        suggestions = []
                        for role, desc in JOB_LIBRARY.items():
                            score = calculate_match(raw_text, desc)
                            suggestions.append((role, score))
                        
                        suggestions.sort(key=lambda x: x[1], reverse=True)
                        for r, s in suggestions[:3]:
                            st.write(f"- **{r}**: {s}% Alignment")
        else:
            st.error("Please paste a job description to calculate your score.")
    else:
        st.error("Please upload a resume file to begin.")

# Sidebar Info
st.sidebar.markdown("---")
st.sidebar.title("How it Works")
st.sidebar.write("HireXpert simulates a modern **Applicant Tracking System (ATS)**. It checks if your file is machine-readable and if your skills align with the target employer's needs.")
