import streamlit as st
import streamlit as st
import google.generativeai as genai
import pypdf 
import docx
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
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

# Initialize Session States to keep track of navigation, inputs, and selected template
if "page" not in st.session_state:
    st.session_state.page = "home"
if "jobs" not in st.session_state:
    st.session_state.jobs = []
if "education" not in st.session_state:
    st.session_state.education = []
if "personal_info" not in st.session_state:
    st.session_state.personal_info = {"name": "", "email": "", "phone": "", "linkedin": "", "skills": ""}
if "template" not in st.session_state:
    st.session_state.template = "Classic Corporate"

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
# 📊 SMART ATS SCORING: Falls back to AI evaluation if the JD is too short
def calculate_score(resume, job_desc):
    if not resume.strip() or not job_desc.strip():
        return 0.0
        
    # If the user only typed a short title/role instead of a full JD
    if len(job_desc.split()) < 10:
        try:
            score_prompt = f"""
            You are an ATS (Applicant Tracking System) algorithm. Compare this resume against the target role title.
            Target Role: {job_desc}
            Resume Content: {resume}
            
            Based on the candidate's skills, industry background, and job titles, calculate an alignment score from 0 to 100.
            Output ONLY a single number representing the percentage (e.g., 78.5). Do not write any letters, words, or symbols.
            """
            ai_score_response = model.generate_content(score_prompt).text.strip()
            cleaned_score = re.findall(r"\d+\.\d+|\d+", ai_score_response)
            if cleaned_score:
                return round(float(cleaned_score[0]), 2)
        except:
            pass # Fall back to traditional matching if API fails

    # Traditional detailed matching for long, text-heavy JDs
    try:
        vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
        tfidf_matrix = vectorizer.fit_transform([resume, job_desc])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return round(similarity[0][0] * 100, 2)
    except:
        return 0.0

# Document Exporter Helper function incorporating 4 Distinct Design Styles
def generate_docx_file():
    doc = Document()
    p_info = st.session_state.personal_info
    template = st.session_state.template
    
    # Configure Typography, Alignment rules, and Hex styling bounds dynamically
    if template == "Classic Corporate":
        font_name = "Times New Roman"
        primary_color = RGBColor(0, 0, 0)        # Deep Charcoal Black
        secondary_color = RGBColor(90, 90, 90)    # Balanced Slate Gray
        header_align = WD_ALIGN_PARAGRAPH.CENTER
    elif template == "Modern Minimalist":
        font_name = "Arial"
        primary_color = RGBColor(44, 62, 80)     # Industrial Midnight Navy
        secondary_color = RGBColor(127, 140, 141) # Cool Muted Platinum 
        header_align = WD_ALIGN_PARAGRAPH.LEFT
    elif template == "Tech Startup":
        font_name = "Calibri"
        primary_color = RGBColor(26, 188, 156)   # Modern Teal Highlight Accent
        secondary_color = RGBColor(52, 73, 94)    # Graphite Dark Base
        header_align = WD_ALIGN_PARAGRAPH.LEFT
    elif template == "Executive Elegance":
        font_name = "Georgia"
        primary_color = RGBColor(139, 0, 0)      # Vintage Burgundy Crimson
        secondary_color = RGBColor(60, 60, 60)    # Soft Obsidian Smoke
        header_align = WD_ALIGN_PARAGRAPH.RIGHT

    # Inject default formatting constraints across basic style rules
    style = doc.styles['Normal']
    style.font.name = font_name
    style.font.size = Pt(11)

    # Header Construction Blocks
    name_p = doc.add_paragraph()
    name_p.alignment = header_align
    name_run = name_p.add_run(p_info['name'] if p_info['name'] else "Your Identity Profile")
    name_run.font.size = Pt(24)
    name_run.bold = True
    name_run.font.color.rgb = primary_color

    info_p = doc.add_paragraph()
    info_p.alignment = header_align
    info_run = info_p.add_run(f"Email: {p_info['email']} | Phone: {p_info['phone']} | LinkedIn: {p_info['linkedin']}")
    info_run.font.size = Pt(10)
    info_run.font.color.rgb = secondary_color
    
    # Skills Structure Node
    if p_info['skills']:
        h = doc.add_heading(level=1)
        h.alignment = header_align if template == "Classic Corporate" else WD_ALIGN_PARAGRAPH.LEFT
        hrun = h.add_run('Core Competencies & Skills')
        hrun.font.color.rgb = primary_color
        hrun.font.size = Pt(14)
        
        p = doc.add_paragraph(p_info['skills'])
        p.paragraph_format.space_after = Pt(12)
        
    # Professional Work History Matrix Engine
    if st.session_state.jobs:
        h = doc.add_heading(level=1)
        h.alignment = header_align if template == "Classic Corporate" else WD_ALIGN_PARAGRAPH.LEFT
        hrun = h.add_run('Professional Experience')
        hrun.font.color.rgb = primary_color
        hrun.font.size = Pt(14)
        
        for job in st.session_state.jobs:
            p = doc.add_paragraph()
            title_run = p.add_run(f"{job['title']} — {job['company']}\n")
            title_run.bold = True
            title_run.font.color.rgb = secondary_color
            
            meta_run = p.add_run(f"{job['dates']} | {job['location']}\n")
            meta_run.font.size = Pt(9.5)
            meta_run.italic = True
            
            # Map structural breaklines back to atomic word document list bullets cleanly
            bullets = job['bullets'].split('\n')
            for bullet in bullets:
                if bullet.strip():
                    bp = doc.add_paragraph(style='List Bullet')
                    bp.add_run(bullet.replace('-', '').strip())
            
            p.paragraph_format.space_after = Pt(12)
            
    # Educational Qualifications Block Matrix
    if st.session_state.education:
        h = doc.add_heading(level=1)
        h.alignment = header_align if template == "Classic Corporate" else WD_ALIGN_PARAGRAPH.LEFT
        hrun = h.add_run('Education Qualifications')
        hrun.font.color.rgb = primary_color
        hrun.font.size = Pt(14)
        
        for edu in st.session_state.education:
            p = doc.add_paragraph()
            edu_run = p.add_run(f"{edu['degree']} — {edu['school']}\n")
            edu_run.bold = True
            
            details_run = p.add_run(f"Graduation Status: {edu['date']} | Reference Benchmarks: {edu['details']}")
            details_run.font.size = Pt(10)
            p.paragraph_format.space_after = Pt(8)
            
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
# SCREEN 3: RESUME BUILDER (DYNAMIC MULTI-ENTRY ENGINE WITH TEMPLATE CONTROL)
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
        with st.form("education_form", clear_on_submit=True):
            school_name = st.text_input("Institution / University Name", placeholder="e.g., Stanford University")
            degree_name = st.text_input("Degree / Major", placeholder="e.g., Bachelor of Science in Computer Science")
            grad_date = st.text_input("Graduation Date / Timeline", placeholder="e.g., May 2024")
            edu_details = st.text_input("Additional Info (GPA, Honors, etc.)", placeholder="e.g., CGPA 3.9/4.0")
            
            if st.form_submit_button("➕ Add Education Entry"):
                if school_name and degree_name:
                    st.session_state.education.append({
                        "school": school_name, "degree": degree_name, "date": grad_date, "details": edu_details
                    })
                    st.success(f"Added entry for {degree_name}!")
                else:
                    st.error("Institution and Degree details are mandatory fields.")

        st.markdown("---")
        
        # FORM 3: PROFESSIONAL EXPERIENCE WITH GENERATOR
        st.subheader("💼 Section 3: Professional Experience")
        with st.form("experience_form", clear_on_submit=True):
            job_title = st.text_input("Job Title / Position", placeholder="e.g., Senior Systems Engineer")
            company = st.text_input("Company / Organization", placeholder="e.g., Google Inc.")
            dates = st.text_input("Employment Dates / Timeline", placeholder="e.g., June 2022 - Present")
            location = st.text_input("Location", placeholder="e.g., San Francisco, CA")
            raw_description = st.text_area("Responsibilities Summary (For AI expansion)", placeholder="Type rough notes here. The system will optimize it into professional impact bullet points.")
            
            if st.form_submit_button("➕ Add Job Profile Entry"):
                if job_title and company:
                    if raw_description.strip():
                        with st.spinner("Refining job descriptions using semantic optimization models..."):
                            bullet_prompt = f"Convert these casual job responsibility descriptions into 3 strong, result-oriented, metrics-driven bullet points for an ATS resume. Do not include introductory remarks or pleasantries. Output ONLY raw bullet points starting with standard dash marks (-). Text: {raw_description}"
                            refined_bullets = model.generate_content(bullet_prompt).text.strip()
                    else:
                        refined_bullets = "- Managed assigned roles and operational performance benchmarks."
                        
                    st.session_state.jobs.append({
                        "title": job_title, "company": company, "dates": dates, "location": location, "bullets": refined_bullets
                    })
                    st.success(f"Successfully processed and stored historical entry for {job_title}!")
                else:
                    st.error("Position Title and Company Name cannot be left blank.")

    # RIGHT COLUMN: LIVE DATA VIEWPORT, TEMPLATE ENGINE & INTERACTIVE DOWNLOADER
    with col2:
        st.subheader("🎨 Select Resume Layout Template")
        st.session_state.template = st.selectbox(
            "Choose a layout template design strategy:",
            ["Classic Corporate", "Modern Minimalist", "Tech Startup", "Executive Elegance"]
        )
        
        st.markdown(f"**Current Layout Profile Active:** `{st.session_state.template}`")
        st.markdown("---")
        st.subheader("👁️ Live Profile Mockup Preview")
        
        # Display Current Status Summary
        st.markdown(f"### **{st.session_state.personal_info['name'] if st.session_state.personal_info['name'] else 'Your Profile Name'}**")
        st.write(f"📧 {st.session_state.personal_info['email']} | 📱 {st.session_state.personal_info['phone']} | 🔗 {st.session_state.personal_info['linkedin']}")
        
        if st.session_state.personal_info['skills']:
            st.markdown("#### **Skills & Core Strengths**")
            st.write(st.session_state.personal_info['skills'])
            
        if st.session_state.jobs:
            st.markdown("#### **Professional Record Workspace**")
            for idx, job in enumerate(st.session_state.jobs):
                st.markdown(f"**{job['title']}** at *{job['company']}* ({job['dates']}) — `{job['location']}`")
                st.write(job['bullets'])
                if st.button(f"🗑️ Delete Entry {idx+1}", key=f"del_job_{idx}"):
                    st.session_state.jobs.pop(idx)
                    st.rerun()
                    
        if st.session_state.education:
            st.markdown("#### **Academic Credentials History**")
            for idx, edu in enumerate(st.session_state.education):
                st.markdown(f"**{edu['degree']}** — *{edu['school']}* ({edu['date']})")
                st.caption(edu['details'])
                if st.button(f"🗑️ Remove Academic Entry {idx+1}", key=f"del_edu_{idx}"):
                    st.session_state.education.pop(idx)
                    st.rerun()
                    
        st.markdown("---")
        st.subheader("📥 Export Final Draft Asset")
        
        if st.session_state.personal_info['name'] or st.session_state.jobs or st.session_state.education:
            docx_buffer = generate_docx_file()
            st.download_button(
                label=f"📥 Download Structured {st.session_state.template} (.docx)",
                data=docx_buffer,
                file_name=f"HireXpert_{st.session_state.template.replace(' ', '_')}_{st.session_state.personal_info['name'].replace(' ', '_') if st.session_state.personal_info['name'] else 'Draft'}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
        else:
            st.warning("Ensure at least one informational component contains validated fields to enable file exports.")
