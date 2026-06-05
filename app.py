import streamlit as st
import google.generativeai as genai
import pypdf 
import docx
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

# 🔑 Load Gemini API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# 🤖 Load model
model = genai.GenerativeModel("gemini-2.5-flash")

# 🎨 Page Config
st.set_page_config(page_title="HireXpert", page_icon="🤖", layout="wide")

# 🔄 Initialize Session States for multi-entry management
if "page" not in st.session_state:
    st.session_state.page = "home"
if "jobs" not in st.session_state:
    st.session_state.jobs = []
if "education" not in st.session_state:
    st.session_state.education = []

# 📄 Extraction Helper Function
def extract_text(uploaded_file):
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = pypdf.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs if para.text])
        elif uploaded_file.type == "text/plain":
            uploaded_file.seek(0) 
            text = uploaded_file.read().decode("utf-8")
        else:
            return None
    except Exception as e:
        print(f"Extraction error: {e}") 
        return None
    return text.strip()

# 📊 ATS Scoring Function
def calculate_score(resume, job_desc):
    if not resume.strip() or not job_desc.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
        tfidf_matrix = vectorizer.fit_transform([resume, job_desc])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return round(similarity * 100, 2)
    except:
        return 0.0

# 📥 Dynamic Resume Document Builder (.docx) Helper
def generate_docx(full_name, email, phone, linkedin, skills, jobs_list, edu_list):
    doc = docx.Document()
    doc.add_heading(full_name if full_name else "Resume", level=0)
    doc.add_paragraph(f"Email: {email} | Phone: {phone} | LinkedIn: {linkedin}")
    
    # Loop over all recorded jobs
    doc.add_heading("Experience", level=1)
    for job in jobs_list:
        p_exp = doc.add_paragraph()
        p_exp.add_run(f"{job['title']} \n").bold = True
        p_exp.add_run(f"{job['company']} ({job['duration']}) — {job['years']} Years Exp\n").italic = True
        p_exp.add_run(job['bullets'])
    
    # Loop over all recorded degrees
    doc.add_heading("Education", level=1)
    for edu in edu_list:
        p_edu = doc.add_paragraph()
        p_edu.add_run(f"{edu['degree']}\n").bold = True
        p_edu.add_run(f"{edu['college']} (Class of {edu['year']})")
    
    doc.add_heading("Skills", level=1)
    doc.add_paragraph(skills)
    
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio


# ==========================================
# 🏠 OPTION 1: THE WELCOME LANDING PAGE
# ==========================================
if st.session_state.page == "home":
    st.title("🤖 Welcome to HireXpert")
    st.write("What would you like to do today?")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Analyze Existing Resume")
        st.write("Upload a resume and match it against a Job Description using semantic AI analytics.")
        if st.button("🚀 Go to Analyzer", use_container_width=True):
            st.session_state.page = "analyzer"
            st.rerun()
            
    with col2:
        st.subheader("📄 Build a New Resume")
        st.write("Generate a completely new, structurally clean, ATS-compliant professional profile.")
        if st.button("🛠️ Go to Builder", use_container_width=True):
            st.session_state.page = "builder"
            st.rerun()


# ==========================================
# 📊 OPTION 2: THE ANALYZER INTERFACE
# ==========================================
elif st.session_state.page == "analyzer":
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"
        st.rerun()
        
    st.title("🤖 HireXpert - Resume Analyzer")
    
    resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
    job_desc_input = st.text_area("Paste Job Description")

    if st.button("Analyze Resume"):
        if resume_file is None or job_desc_input.strip() == "":
            st.warning("Please upload resume and enter job description")
        else:
            resume_text = extract_text(resume_file)
            if not resume_text:
                st.error("❌ This file is empty, corrupted, or unsupported.")
                st.stop() 
                
            score = calculate_score(resume_text, job_desc_input)
            st.subheader("📊 ATS Score")
            st.progress(score / 100.0)
            st.write(f"**Score: {score}%**")

            current_date_str = datetime.now().strftime("%B %Y")
            prompt = f"""
            You are an expert technical recruiter analyzing a resume against a target Job Description.
            CRITICAL CONTEXT: The current actual date is exactly {current_date_str}. Do NOT treat current dates as future items.

            Please analyze this profile comprehensively:
            - Strengths 
            - Weaknesses
            - Missing keywords
            - Improvement suggestions

            Resume: {resume_text}
            Job Description: {job_desc_input}
            """

            with st.spinner("Analyzing with AI..."):
                response = model.generate_content(prompt)
                st.subheader("🤖 AI Feedback")
                st.write(response.text)


# ==========================================
# 📄 OPTION 3: THE BUILDER INTERFACE
# ==========================================
elif st.session_state.page == "builder":
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"
        st.rerun()
        
    st.title("📄 Multi-History AI Resume Builder")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Personal Information")
        full_name = st.text_input("Full Name", placeholder="Jane Doe")
        email = st.text_input("Email Address", placeholder="janedoe@email.com")
        phone = st.text_input("Phone Number", placeholder="+91 98765 43210")
        linkedin = st.text_input("LinkedIn Profile URL")
        skills = st.text_area("Skills (Comma-separated)", placeholder="Team Management, Curriculum Design, ERP Software")

        st.subheader("🎓 Education History")
        with st.form("edu_form", clear_on_submit=True):
            deg = st.text_input("Degree / Qualification", placeholder="B.Ed or M.Sc Chemistry")
            inst = st.text_input("College / School Name")
            yr = st.text_input("Graduation Year", placeholder="1998")
            if st.form_submit_button("➕ Add This Education Block"):
                if deg and inst:
                    st.session_state.education.append({"degree": deg, "college": inst, "year": yr})
                    st.success(f"Added {deg}!")
                else:
                    st.warning("Please fill out Degree and College details.")

    with col2:
        st.subheader("💼 Work Experience History")
        with st.form("job_form", clear_on_submit=True):
            j_title = st.text_input("Job Title / Role", placeholder="Senior Teacher or Operations Executive")
            comp = st.text_input("Company / School Name", placeholder="TVS / Public School")
            dur = st.text_input("Duration Timelines", placeholder="2015 - 2021")
            y_exp = st.text_input("Years of Experience in this specific role", placeholder="6")
            
            if st.form_submit_button("✨ Add & Generate AI Responsibilities"):
                if j_title and y_exp:
                    with st.spinner("AI is building professional achievements..."):
                        ai_prompt = f"""
                        Write a professional, ATS-optimized list of resume bullet points for this specific position:
                        Role: {j_title}
                        Institution/Company: {comp}
                        Experience Length: {y_exp} years

                        Instructions:
                        - Write exactly 4 bullet points using impactful action verbs suited for this career path.
                        - Do NOT add greeting texts or generic wrappers. Just output raw '• ' bullets.
                        """
                        ai_response = model.generate_content(ai_prompt)
                        # Append the complete structured dict into list array memory
                        st.session_state.jobs.append({
                            "title": j_title, "company": comp, 
                            "duration": dur, "years": y_exp, 
                            "bullets": ai_response.text
                        })
                        st.success(f"Successfully saved and generated role data for {j_title}!")
                else:
                    st.warning("Please enter Job Title and Years of Experience.")

    # Reset Buttons to clear entries if mistakes are made
    if st.button("🧹 Clear All Entered Jobs & Degrees"):
        st.session_state.jobs = []
        st.session_state.education = []
        st.rerun()

    st.markdown("---")
    st.subheader("👀 Live Resume Preview")
    
    # Core structural builder visualization
    preview_markdown = f"""
    # {full_name if full_name else 'Your Name'}
    📧 {email} | 📱 {phone} | 🔗 {linkedin}
    ---
    ### 💼 Experience History
    """
    
    for job in st.session_state.jobs:
