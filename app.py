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

# Initialize Session States to keep track of navigation and inputs safely
if "page" not in st.session_state:
    st.session_state.page = "home"
if "jobs" not in st.session_state:
    st.session_state.jobs = []
if "education" not in st.session_state:
    st.session_state.education = []
if "personal_info" not in st.session_state:
    st.session_state.personal_info = {"name": "", "email": "", "phone": "", "linkedin": "", "skills": ""}

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
def generate_docx(p_info, jobs_list, edu_list):
    doc = docx.Document()
    doc.add_heading(p_info['name'] if p_info['name'] else "Resume", level=0)
    doc.add_paragraph(f"Email: {p_info['email']} | Phone: {p_info['phone']} | LinkedIn: {p_info['linkedin']}")
    
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
    doc.add_paragraph(p_info['skills'])
    
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
    st.write("Fill each section and press its specific save button when done.")
    
    col1, col2 = st.columns(2)
    with col1:
        # FORM 1: PERSONAL DETAILS SECTION
        st.subheader("👤 Section 1: Personal Information")
        with st.form("personal_form", clear_on_submit=False):
            f_name = st.text_input("Full Name", value=st.session_state.personal_info["name"], placeholder="Jane Doe")
            e_mail = st.text_input("Email Address", value=st.session_state.personal_info["email"], placeholder="janedoe@email.com")
            p_hone = st.text_input("Phone Number", value=st.session_state.personal_info["phone"], placeholder="+91 98765 43210")
            l_inkedin = st.text_input("LinkedIn Profile URL", value=st.session_state.personal_info["linkedin"])
            s_kills = st.text_area("Skills (Comma-separated)", value=st.session_state.personal_info["skills"], placeholder="Curriculum Mapping, Operations management")
            
            if st.form_submit_button("💾 Save Personal Info & Skills"):
                st.session_state.personal_info = {
                    "name": f_name, "email": e_mail, "phone": p_hone, "linkedin": l_inkedin, "skills": s_kills
                }
                st.success("Personal information lock-saved!")

        st.markdown("---")

        # FORM 2: EDUCATION SECTION WITH SPELLING CORRECTION
        st.subheader("🎓 Section 2: Education Qualifications")
        with st.form("edu_form", clear_on_submit=True):
            deg = st.text_input("Degree / Qualification Title", placeholder="B.Ed Education or M.Sc Chemistry")
            inst = st.text_input("Institution / University Name", placeholder="Madras University")
            yr = st.text_input("Graduation Class Year", placeholder="1998")
            
            if st.form_submit_button("➕ Save & Spell-Check Education Block"):
                if deg and inst:
                    with st.spinner("AI checking spelling of degree & university names..."):
                        spell_prompt = f"Verify and correct any spelling errors in this Degree: '{deg}' and Institution: '{inst}'. Respond ONLY with the corrected details in this exact layout format: Corrected Degree | Corrected Institution. Do not include any other text."
                        spell_response = model.generate_content(spell_prompt).text.strip()
                        
                        try:
                            corrected_deg, corrected_inst = spell_response.split("|")
                            deg = corrected_deg.strip()
                            inst = corrected_inst.strip()
                        except:
                            pass # Fallback to original inputs if split fails
                        
                    st.session_state.education.append({"degree": deg, "college": inst, "year": yr})
                    st.success(f"Added verified entry: {deg} from {inst}!")
                else:
                    st.warning("Please enter Qualification and Institution entries.")

    with col2:
        # FORM 3: CAREER HISTORY WITH DEEP ACHIEVEMENTS GENERATION
        st.subheader("💼 Section 3: Career History Blocks")
        with st.form("job_form", clear_on_submit=True):
            j_title = st.text_input("Job Title / Designation", placeholder="Senior Teacher / Operations Specialist")
            comp = st.text_input("Company / Corporate School Name", placeholder="TVS Group / Public School")
            dur = st.text_input("Tenure Duration Timeline", placeholder="2012 - 2018")
            y_exp = st.text_input("Total Years deployed in this track", placeholder="6")
            
            if st.form_submit_button("✨ Save & Auto-Generate AI Achievements"):
                if j_title and y_exp:
                    with st.spinner("AI is calculating deep corporate achievements..."):
                        ai_prompt = f"""
                        Write a highly detailed, professional, broad achievements summary list for a resume.
                        Role Designation: {j_title}
                        Institution/Company Entity: {comp}
                        Years Spent: {y_exp} years

                        Execution Instructions:
                        - Generate exactly 4 detailed bullet points highlighting deep operational milestones, classroom governance, or corporate administration depending on the track.
                        - Focus on broad impacts (e.g., student grade improvements for teachers, supply chain efficiencies for corporate roles).
                        - Output purely raw bullet items starting directly with '• ' characters. No intro wrappers.
                        """
                        ai_response = model.generate_content(ai_prompt)
                        st.session_state.jobs.append({
                            "title": j_title, "company": comp, 
                            "duration": dur, "years": y_exp, 
                            "bullets": ai_response.text
                        })
                        st.success(f"Processed long-form career summary layout for {j_title}!")
                else:
                    st.warning("Please complete Designation and Years fields.")

    if st.button("🧹 Clear All Entered Data Logs"):
        st.session_state.jobs = []
        st.session_state.education = []
        st.session_state.personal_info = {"name": "", "email": "", "phone": "", "linkedin": "", "skills": ""}
        st.rerun()

    st.markdown("---")
    st.subheader("👀 Real-Time Live Profile Layout Preview")
    
    p_info = st.session_state.personal_info
    st.markdown(f"# {p_info['name'] if p_info['name'] else 'Candidate Profile Outline'}")
    st.write(f"📧 {p_info['email']} | 📱 {p_info['phone']} | 🔗 {p_info['linkedin']}")
    st.markdown("---")
    
    st.markdown("### 💼 Professional Work Timeline")
    if not st.session_state.jobs:
        st.caption("No career modules populated yet.")
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
    if p_info['skills']:
        st.code(p_info['skills'], language="text")

    if p_info['name'] and (st.session_state.jobs or st.session_state.education):
        docx_data = generate_docx(p_info, st.session_state.jobs, st.session_state.education)
        st.download_button(
            label="📥 Download Clean ATS Resume Document (.docx)",
            data=docx_data,
            file_name=f"{p_info['name'].replace(' ', '_')}_Resume.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

st.markdown("---")
st.markdown("Made with ❤️ by a student")
