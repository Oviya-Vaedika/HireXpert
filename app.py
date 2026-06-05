import streamlit as st
import google.generativeai as genai
import pypdf 
import docx
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

# Load Gemini API key from Streamlit Secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load the model instance
model = genai.GenerativeModel("gemini-2.5-flash")

# App Page Layout Setup
st.set_page_config(page_title="HireXpert", page_icon="🤖", layout="wide")

# Initialize Session States to keep track of navigation and inputs
if "page" not in st.session_state:
    st.session_state.page = "home"
if "jobs" not in st.session_state:
    st.session_state.jobs = []
if "education" not in st.session_state:
    st.session_state.education = []

# Extraction Helper Function for Analyzer Page
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

# Advanced ATS Semantic Vector Distance Matching
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

# Dynamic Resume Generator Engine (.docx builder)
def generate_docx(full_name, email, phone, linkedin, skills, jobs_list, edu_list):
    doc = docx.Document()
    doc.add_heading(full_name if full_name else "Resume", level=0)
    doc.add_paragraph(f"Email: {email} | Phone: {phone} | LinkedIn: {linkedin}")
    
    doc.add_heading("Experience", level=1)
    for job in jobs_list:
        p_exp = doc.add_paragraph()
        p_exp.add_run(f"{job['title']} \n").bold = True
        p_exp.add_run(f"{job['company']} ({job['duration']}) — {job['years']} Years Exp\n").italic = True
        p_exp.add_run(job['bullets'])
    
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

# SCREEN 1: INTERACTIVE LANDING MENU
if st.session_state.page == "home":
    st.title("🤖 Welcome to HireXpert")
    st.write("What would you like to do today?")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Analyze Existing Resume")
        st.write("Upload a resume and check metrics against target roles using semantic AI models.")
        if st.button("🚀 Go to Analyzer", use_container_width=True):
            st.session_state.page = "analyzer"
            st.rerun()
            
    with col2:
        st.subheader("📄 Build a New Resume")
        st.write("Generate an ATS-compliant, multi-history professional profile with automated AI bullet points.")
        if st.button("🛠️ Go to Builder", use_container_width=True):
            st.session_state.page = "builder"
            st.rerun()
# SCREEN 2: RESUME ANALYZER (SCANNER MODULE)
elif st.session_state.page == "analyzer":
    if st.button("⬅️ Back to Home Menu"):
        st.session_state.page = "home"
        st.rerun()
        
    st.title("🤖 HireXpert - Resume Analyzer")
    st.write("Evaluate how well a profile fits given criteria parameters.")
    
    resume_file = st.file_uploader("Upload Resume File", type=["pdf", "docx", "txt"])
    job_desc_input = st.text_area("Paste Target Profile Criteria / Job Description")

    if st.button("Run Profile Analysis"):
        if resume_file is None or job_desc_input.strip() == "":
            st.warning("Please upload a resume file and copy criteria text into the tracking window.")
        else:
            resume_text = extract_text(resume_file)
            if not resume_text:
                st.error("❌ This file is empty, corrupted, or unsupported.")
                st.stop() 
                
            score = calculate_score(resume_text, job_desc_input)
            st.subheader("📊 Semantic Fit Matrix")
            st.progress(score / 100.0)
            st.write(f"**Calculated Score: {score}%**")

            current_date_str = datetime.now().strftime("%B %Y")
            prompt = f"Analyze resume matching against Job Description. Target date environment is {current_date_str}. Do not flag ongoing profiles up to this date as gaps. Give detailed strengths, weaknesses, missing keywords, and suggestions. Resume: {resume_text} JD: {job_desc_input}"

            with st.spinner("AI Engine is scanning files..."):
                response = model.generate_content(prompt)
                st.subheader("🤖 AI Recruiter Feedback Summary")
                st.write(response.text)

# SCREEN 3: RESUME BUILDER (DYNAMIC MULTI-ENTRY ENGINE)
elif st.session_state.page == "builder":
    if st.button("⬅️ Back to Home Menu"):
        st.session_state.page = "home"
        st.rerun()
        
    st.title("📄 Multi-History AI Resume Builder")
    st.write("Generate custom roles dynamically (e.g., Teaching background, TVS corporate records, etc.).")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Personal Information")
        full_name = st.text_input("Full Name", placeholder="Jane Doe")
        email = st.text_input("Email Address", placeholder="janedoe@email.com")
        phone = st.text_input("Phone Number", placeholder="+91 98765 43210")
        linkedin = st.text_input("LinkedIn Profile URL")
        skills = st.text_area("Skills (Comma-separated lines)", placeholder="Curriculum Mapping, Strategic Planning, Core Python")

        st.subheader("🎓 Education Qualifications")
        with st.form("edu_form", clear_on_submit=True):
            deg = st.text_input("Degree / Qualification Title", placeholder="B.Ed Education or M.Sc Chemistry")
            inst = st.text_input("Institution / University Name")
            yr = st.text_input("Graduation Class Year", placeholder="1998")
            if st.form_submit_button("➕ Append Education Entry"):
                if deg and inst:
                    st.session_state.education.append({"degree": deg, "college": inst, "year": yr})
                    st.success(f"Added entry for {deg} into buffer!")
                else:
                    st.warning("Please complete Qualification and Institution name entries.")

    with col2:
        st.subheader("💼 Career History Blocks")
        with st.form("job_form", clear_on_submit=True):
            j_title = st.text_input("Job Title / Designation", placeholder="Senior Academic Instructor / Operations Analyst")
            comp = st.text_input("Company / Corporate Institution Name", placeholder="Public School / TVS Group")
            dur = st.text_input("Tenure Duration Timeline", placeholder="2012 - 2018")
            y_exp = st.text_input("Total Years deployed in this track", placeholder="6")
            
            if st.form_submit_button("✨ Save & Auto-Generate AI Achievements"):
                if j_title and y_exp:
                    with st.spinner("AI is calculating role accomplishments..."):
                        ai_prompt = f"Write professional, single-column ATS-friendly resume bullets starting directly with symbol characters for a candidate with Title: {j_title}, Company: {comp}, Years: {y_exp}. Only provide the raw points, no greeting text or summary introduction elements."
                        ai_response = model.generate_content(ai_prompt)
                        st.session_state.jobs.append({
                            "title": j_title, "company": comp, 
                            "duration": dur, "years": y_exp, 
                            "bullets": ai_response.text
                        })
                        st.success(f"Successfully processed AI records layout for {j_title}!")
                else:
                    st.warning("Please verify Job Designation and Total Years parameters before submitting.")

    if st.button("🧹 Clear All Entered Logs"):
        st.session_state.jobs = []
        st.session_state.education = []
        st.rerun()

    st.markdown("---")
    st.subheader("👀 Real-Time Live Profile Layout Preview")
    
    st.markdown(f"# {full_name if full_name else 'Candidate Profile Outline'}")
    st.write(f"📧 {email} | 📱 {phone} | 🔗 {linkedin}")
    st.markdown("---")
    
    st.markdown("### 💼 Professional Work Timeline")
    if not st.session_state.jobs:
        st.caption("No corporate or school career modules populated yet.")
    else:
        for job in st.session_state.jobs:
            st.markdown(f"**{job['title']}** at *{job['company']}* ({job['duration']}) — {job['years']} Years Active")
            st.write(job['bullets'])
            st.markdown("")
        
    st.markdown("### 🎓 Academic Qualifications Timeline")
    if not st.session_state.education:
        st.caption("No structural graduation lines recorded.")
    else:
        for edu in st.session_state.education:
            st.markdown(f"• **{edu['degree']}** — *{edu['college']}* (Class of {edu['year']})")
        
    st.markdown("### 🛠️ Core Functional Competencies")
    if skills:
        st.code(skills, language="text")

    if full_name and (st.session_state.jobs or st.session_state.education):
        docx_data = generate_docx(
            full_name, email, phone, linkedin, skills, 
            st.session_state.jobs, st.session_state.education
        )
        st.download_button(
            label="📥 Download Clean ATS Resume Document (.docx)",
            data=docx_data,
            file_name=f"{full_name.replace(' ', '_')}_Resume.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

st.markdown("---")
st.markdown("Made with ❤️ by a student")
