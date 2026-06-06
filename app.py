import streamlit as st
import google.generativeai as genai
import pypdf 
import docx
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
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

# Initialize Session States safely for 6 Core Modules
if "page" not in st.session_state:
    st.session_state.page = "home"
if "jobs" not in st.session_state:
    st.session_state.jobs = []
if "education" not in st.session_state:
    st.session_state.education = []
if "projects" not in st.session_state:
    st.session_state.projects = []
if "personal_info" not in st.session_state:
    st.session_state.personal_info = {"name": "", "email": "", "phone": "", "linkedin": ""}
if "skills_info" not in st.session_state:
    st.session_state.skills_info = {"skills": ""}
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
            pass 

    try:
        vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
        tfidf_matrix = vectorizer.fit_transform([resume, job_desc])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return round(similarity * 100, 2)
    except:
        return 0.0
# Helper to set cell background colors for advanced modern layouts
def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def generate_docx_file():
    doc = Document()
    p_info = st.session_state.personal_info
    s_info = st.session_state.skills_info
    template = st.session_state.template
    
    # ----------------------------------------------------
    # DESIGN STRATEGY 1: MODERN TWO-COLUMN SIDEBAR
    # ----------------------------------------------------
    if template == "Modern Minimalist":
        style = doc.styles['Normal']
        style.font.name = 'Arial'
        style.font.size = Pt(10)
        
        table = doc.add_table(rows=1, cols=2)
        table.autofit = False
        table.columns[0].width = Inches(2.2) # Left Sidebar Column
        table.columns[1].width = Inches(4.3) # Main Body Content
        
        left_cell = table.cell(0, 0)
        right_cell = table.cell(0, 1)
        set_cell_background(left_cell, "F2F4F4")
        
        # Module 1 & 5: Sidebar Personal Info & Technical Skills
        lp = left_cell.paragraphs[0]
        ln_run = lp.add_run(f"{p_info['name'] if p_info['name'] else 'Profile name'}\n\n")
        ln_run.bold = True
        ln_run.font.size = Pt(16)
        ln_run.font.color.rgb = RGBColor(44, 62, 80)
        
        lp.add_run(f"📱 {p_info['phone']}\n✉️ {p_info['email']}\n🔗 {p_info['linkedin']}\n\n")
        if s_info['skills']:
            sk_head = left_cell.add_paragraph().add_run("EXPERTISE & SKILLS\n")
            sk_head.bold = True
            left_cell.add_paragraph(s_info['skills'].replace(',', '\n'))
            
        # Module 2, 3 & 4: Main Column Content
        if st.session_state.jobs:
            right_cell.add_paragraph().add_run("PROFESSIONAL EXPERIENCE").bold = True
            for job in st.session_state.jobs:
                jp = right_cell.add_paragraph()
                jp.add_run(f"▪ {job['title']} — {job['company']} ({job['dates']})\n").bold = True
                bullets = job['bullets'].split('\n')
                for b in bullets:
                    if b.strip():
                        right_cell.add_paragraph(style='List Bullet').add_run(b.replace('-', '').strip())
                        
        if st.session_state.projects:
            right_cell.add_paragraph().add_run("\nPROJECTS & INTERNSHIPS").bold = True
            for proj in st.session_state.projects:
                pp = right_cell.add_paragraph()
                pp.add_run(f"🚀 {proj['title']} ({proj['timeline']})\n").bold = True
                pp.add_run(proj['details'])

        if st.session_state.education:
            right_cell.add_paragraph().add_run("\nEDUCATION QUALIFICATIONS").bold = True
            for edu in st.session_state.education:
                ep = right_cell.add_paragraph()
                ep.add_run(f"🎓 {edu['degree']} from {edu['school']} ({edu['date']})")

    # ----------------------------------------------------
    # DESIGN STRATEGY 2: TECH STARTUP TOP BANNER ACCENT
    # ----------------------------------------------------
    elif template == "Tech Startup":
        style = doc.styles['Normal']
        style.font.name = 'Calibri'
        style.font.size = Pt(11)
        
        banner_table = doc.add_table(rows=1, cols=1)
        cell = banner_table.cell(0, 0)
        set_cell_background(cell, "1ABC9C")
        
        bp = cell.paragraphs[0]
        bp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        n_run = bp.add_run(p_info['name'] if p_info['name'] else "Developer Profile")
        n_run.font.size = Pt(22)
        n_run.bold = True
        n_run.font.color.rgb = RGBColor(255, 255, 255)
        
        mp = doc.add_paragraph()
        mp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        mp.add_run(f"{p_info['email']} | {p_info['phone']} | {p_info['linkedin']}")
        
        if s_info['skills']:
            doc.add_heading("// Technical Core Architecture", level=1).font.color.rgb = RGBColor(52, 73, 94)
            doc.add_paragraph(s_info['skills'])
            
        if st.session_state.jobs:
            doc.add_heading("// Professional Deployments", level=1).font.color.rgb = RGBColor(52, 73, 94)
            for job in st.session_state.jobs:
                doc.add_paragraph(f"{job['title']} @ {job['company']} ({job['dates']})").bold = True
                bullets = job['bullets'].split('\n')
                for b in bullets:
                    if b.strip():
                        doc.add_paragraph(style='List Bullet').add_run(b.replace('-', '').strip())

        if st.session_state.projects:
            doc.add_heading("// Practical Implementations & Internships", level=1).font.color.rgb = RGBColor(52, 73, 94)
            for proj in st.session_state.projects:
                p = doc.add_paragraph(f"{proj['title']} — {proj['timeline']}").bold = True
                doc.add_paragraph(proj['details'])

        if st.session_state.education:
            doc.add_heading("// Academic Framework", level=1).font.color.rgb = RGBColor(52, 73, 94)
            for edu in st.session_state.education:
                doc.add_paragraph(f"{edu['degree']} — {edu['school']} ({edu['date']})")

    # ----------------------------------------------------
    # DESIGN STRATEGY 3 & 4: CORPORATE / EXECUTIVE BALANCED STACKS
    # ----------------------------------------------------
    else:
        is_exec = (template == "Executive Elegance")
        font_choice = "Georgia" if is_exec else "Times New Roman"
        color_choice = RGBColor(139, 0, 0) if is_exec else RGBColor(0, 0, 0)
        
        style = doc.styles['Normal']
        style.font.name = font_choice
        style.font.size = Pt(11)
        
        np = doc.add_paragraph()
        np.alignment = WD_ALIGN_PARAGRAPH.RIGHT if is_exec else WD_ALIGN_PARAGRAPH.CENTER
        n_run = np.add_run(p_info['name'])
        n_run.font.size = Pt(22)
        n_run.bold = True
        n_run.font.color.rgb = color_choice
        
        ip = doc.add_paragraph()
        ip.alignment = WD_ALIGN_PARAGRAPH.RIGHT if is_exec else WD_ALIGN_PARAGRAPH.CENTER
        ip.add_run(f"{p_info['email']} | {p_info['phone']} | {p_info['linkedin']}")
        
        doc.add_paragraph("═" * 55).alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        if s_info['skills']:
            doc.add_paragraph().add_run("CORE COMPETENCIES").bold = True
            doc.add_paragraph(s_info['skills'])
            
        if st.session_state.jobs:
            doc.add_paragraph().add_run("\nCHRONOLOGICAL WORK RECORD").bold = True
            for job in st.session_state.jobs:
                p = doc.add_paragraph()
                p.add_run(f"{job['title']} — {job['company']} ({job['dates']})\n").bold = True
                bullets = job['bullets'].split('\n')
                for b in bullets:
                    if b.strip():
                        doc.add_paragraph(style='List Bullet').add_run(b.replace('-', '').strip())
                        
        if st.session_state.projects:
            doc.add_paragraph().add_run("\nTECHNICAL INITIATIVES & INTERNSHIPS").bold = True
            for proj in st.session_state.projects:
                p = doc.add_paragraph()
                p.add_run(f"{proj['title']} ({proj['timeline']})\n").bold = True
                doc.add_paragraph(proj['details'])

        if st.session_state.education:
            doc.add_paragraph().add_run("\nACADEMIC QUALIFICATIONS").bold = True
            for edu in st.session_state.education:
                doc.add_paragraph(f"• {edu['degree']} from {edu['school']} ({edu['date']})")

    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio
# SCREEN INTERACTION GATEWAY
if st.session_state.page == "home":
    st.title("🤖 Welcome to HireXpert")
    st.write("What would you like to do today?")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Analyze Existing Resume")
        if st.button("🚀 Go to Analyzer", use_container_width=True):
            st.session_state.page = "analyzer"
            st.rerun()
    with col2:
        st.subheader("📄 Build a New Resume")
        if st.button("🛠️ Go to Builder", use_container_width=True):
            st.session_state.page = "builder"
            st.rerun()

elif st.session_state.page == "analyzer":
    if st.button("⬅️ Back to Home Menu"):
        st.session_state.page = "home"
        st.rerun()
    st.title("🤖 HireXpert - Resume Analyzer")
    resume_file = st.file_uploader("Upload Resume File", type=["pdf", "docx", "txt"])
    job_desc_input = st.text_area("Paste Target Profile Criteria / Job Description")
    if st.button("Run Profile Analysis"):
        if resume_file and job_desc_input.strip():
            resume_text = extract_text(resume_file)
            score = calculate_score(resume_text, job_desc_input)
            st.subheader(f"📊 Semantic Fit Score: {score}%")
            st.progress(score / 100.0)
            with st.spinner("Analyzing data layers..."):
                st.write(model.generate_content(f"Analyze: {resume_text} Against Criteria: {job_desc_input}").text)

elif st.session_state.page == "builder":
    if st.button("⬅️ Back to Home Menu"):
        st.session_state.page = "home"
        st.rerun()
        
    st.title("📄 Multi-History AI Resume Builder")
    col1, col2 = st.columns(2)
    
    with col1:
        # MODULE 1: PERSONAL DETAILS
        st.subheader("👤 Personal Details")
        with st.form("personal_form"):
            f_name = st.text_input("Full Name", value=st.session_state.personal_info["name"], placeholder="Jane Doe")
            e_mail = st.text_input("Email Address", value=st.session_state.personal_info["email"], placeholder="jane@example.com")
            p_hone = st.text_input("Phone Number", value=st.session_state.personal_info["phone"], placeholder="+91 98765 43210")
            l_inkedin = st.text_input("LinkedIn Profile URL", value=st.session_state.personal_info["linkedin"])
            if st.form_submit_button("💾 Save Personal Data"):
                st.session_state.personal_info = {"name": f_name, "email": e_mail, "phone": p_hone, "linkedin": l_inkedin}
                st.success("Personal Details saved!")

        # MODULE 2: WORK EXPERIENCE WITH MULTI-APPEND TRIGGER 
        st.subheader("💼 Professional Experience")
        with st.form("experience_form", clear_on_submit=True):
            job_title = st.text_input("Job Title / Position")
            company = st.text_input("Company / Organization")
            dates = st.text_input("Employment Dates / Timeline")
            location = st.text_input("Location")
            raw_desc = st.text_area("Responsibilities Notes (AI Optimizes this)")
            if st.form_submit_button("➕ Add Work Position Profile"):
                if job_title and company:
                    with st.spinner("Optimizing job points using AI metrics..."):
                        bullets = model.generate_content(f"Convert into 3 powerful resume bullet points starting with standard (-): {raw_desc}").text.strip() if raw_desc.strip() else "- Performance managed roles."
                    st.session_state.jobs.append({"title": job_title, "company": company, "dates": dates, "location": location, "bullets": bullets})
                    st.success(f"Added position entry for {job_title}!")
                    st.rerun()

        # MODULE 3: EDUCATION
        st.subheader("🎓 Academic Qualifications")
        with st.form("education_form", clear_on_submit=True):
            school_name = st.text_input("Institution / University Name")
            degree_name = st.text_input("Degree / Major")
            grad_date = st.text_input("Graduation Timeline")
            edu_details = st.text_input("Additional Info (GPA, Awards)")
            if st.form_submit_button("➕ Add Academic History Entry"):
                if school_name and degree_name:
                    st.session_state.education.append({"school": school_name, "degree": degree_name, "date": grad_date, "details": edu_details})
                    st.success("Education entry added!")
                    st.rerun()

        # MODULE 4: PROJECTS OR INTERNSHIPS
        st.subheader("🚀 Projects & Internships")
        with st.form("project_form", clear_on_submit=True):
            p_title = st.text_input("Project / Internship Title", placeholder="e.g., E-Commerce Architecture Backend")
            p_time = st.text_input("Timeline / Duration", placeholder="e.g., Jan 2024 - Mar 2024")
            p_desc = st.text_area("Scope & Technologies Used Summary")
            if st.form_submit_button("➕ Add Project Portfolio Entry"):
                if p_title:
                    st.session_state.projects.append({"title": p_title, "timeline": p_time, "details": p_desc})
                    st.success("Project entry successfully stored!")
                    st.rerun()

        # MODULE 5: SKILLS
        st.subheader("🛠️ Core Competencies & Skills")
        with st.form("skills_form"):
            skills_input = st.text_area("Skills (Comma-separated matrix)", value=st.session_state.skills_info["skills"], placeholder="Python, Django, AWS Cloud Architecture")
            if st.form_submit_button("💾 Save Skills Entry"):
                st.session_state.skills_info = {"skills": skills_input}
                st.success("Skills stored successfully!")

    # MODULE 6: TEMPLATE CONTROL PANEL & VIEWPORT REAL-TIME DRAW WINDOWS
    with col2:
        st.subheader("🎨 Template Strategy Framework")
        st.session_state.template = st.selectbox(
            "Select Layout Style Blueprint (Changes Document Structure)", 
            ["Classic Corporate", "Modern Minimalist", "Tech Startup", "Executive Elegance"]
        )
        
        st.info(f"Blueprint Strategy: **{st.session_state.template}**")
        st.markdown("---")
        st.subheader("👁️ Dynamic Profile Mockup Tracker")
        
        # Display Mockup Summaries live as data saves
        st.markdown(f"### **{st.session_state.personal_info['name'] if st.session_state.personal_info['name'] else 'Your Identity Title'}**")
        st.write(f"✉️ {st.session_state.personal_info['email']} | 📱 {st.session_state.personal_info['phone']} | 🔗 {st.session_state.personal_info['linkedin']}")
        
        if st.session_state.skills_info['skills']:
            st.markdown(f"**Core Expertise:** {st.session_state.skills_info['skills']}")
            
        if st.session_state.jobs:
            st.markdown("#### **Professional Trajectory**")
            for idx, j in enumerate(st.session_state.jobs):
                st.write(f"💼 **{j['title']}** at {j['company']} ({j['dates']})")
                st.caption(j['bullets'])
                if st.button(f"🗑️ Clear Job {idx+1}", key=f"rm_j_{idx}"):
                    st.session_state.jobs.pop(idx)
                    st.rerun()
                    
        if st.session_state.projects:
            st.markdown("#### **Project Initiatives**")
            for idx, p in enumerate(st.session_state.projects):
                st.write(f"🚀 **{p['title']}** ({p['timeline']})")
                st.write(p['details'])
                if st.button(f"🗑️ Clear Project {idx+1}", key=f"rm_p_{idx}"):
                    st.session_state.projects.pop(idx)
                    st.rerun()

        if st.session_state.education:
            st.markdown("#### **Academic History**")
            for idx, e in enumerate(st.session_state.education):
                st.write(f"🎓 **{e['degree']}** — {e['school']} ({e['date']})")
                if st.button(f"🗑️ Clear Education {idx+1}", key=f"rm_e_{idx}"):
                    st.session_state.education.pop(idx)
                    st.rerun()
            
        st.markdown("---")
        if st.session_state.personal_info['name']:
            st.download_button(
                label=f"📥 Download Fully Structured {st.session_state.template} File",
                data=generate_docx_file(),
                file_name=f"HireXpert_{st.session_state.template.replace(' ', '_')}_{st.session_state.personal_info['name'].replace(' ', '_')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
        else:
            st.warning("Complete Box 1 (Personal Details) with validated name attributes to enable document downloads.")
