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

# 🔄 Initialize Page Session State if not already set
if "page" not in st.session_state:
    st.session_state.page = "home"
if "ai_bullet_points" not in st.session_state:
    st.session_state.ai_bullet_points = ""

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

# 📥 Resume Document Builder Helper
def generate_docx(full_name, email, phone, linkedin, skills, job_title, company, duration, job_desc, degree, college, grad_year):
    doc = docx.Document()
    doc.add_heading(full_name if full_name else "Resume", level=0)
    doc.add_paragraph(f"Email: {email} | Phone: {phone} | LinkedIn: {linkedin}")
    
    doc.add_heading("Experience", level=1)
    p_exp = doc.add_paragraph()
    p_exp.add_run(f"{job_title} \n").bold = True
    p_exp.add_run(f"{company} ({duration})\n").italic = True
    p_exp.add_run(job_desc)
    
    doc.add_heading("Education", level=1)
    p_edu = doc.add_paragraph()
    p_edu.add_run(f"{degree}\n").bold = True
    p_edu.add_run(f"{college} (Class of {grad_year})")
    
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
            
            CRITICAL CONTEXT: 
            - The current actual date is exactly {current_date_str}. 
            - Do NOT treat dates matching or preceding {current_date_str} (such as March 2026) as future dates. 
            - A candidate list entry working "Till Now" or "Present" signifies unbroken active tenure.

            Please analyze this profile comprehensively:
            - Strengths (Include tenure, active standing, and key placements like HCL if present)
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
        
    st.title("📄 AI-Powered Resume Builder")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Personal Information")
        full_name = st.text_input("Full Name", placeholder="John Doe")
        email = st.text_input("Email Address", placeholder="johndoe@email.com")
        phone = st.text_input("Phone Number", placeholder="+91 98765 43210")
        linkedin = st.text_input("LinkedIn Profile URL")
        skills = st.text_area("Skills (Comma-separated)", placeholder="Python, SQL, Project Management")

    with col2:
        st.subheader("💼 Work Experience")
        job_title = st.text_input("Your Job Title / Role", placeholder="Software Engineer")
        company = st.text_input("Company Name", placeholder="HCL Technologies")
        duration = st.text_input("Duration", placeholder="June 2022 - Present")
        years_exp = st.text_input("Total Years of Experience in this role", placeholder="3")

        # ✨ The AI Button replacing the old manual description text area
        if st.button("✨ Generate AI Responsibilities", use_container_width=True):
            if not job_title or not years_exp:
                st.warning("Please enter your Job Title and Years of Experience first!")
            else:
                with st.spinner("AI is crafting professional achievements..."):
                    ai_prompt = f"""
                    Write a professional, ATS-optimized list of resume bullet points for a candidate with the following background:
                    Role: {job_title}
                    Company: {company}
                    Tenure Length: {years_exp} years

                    Instructions:
                    - Write exactly 4-5 bullet points.
                    - Start each bullet point with a strong action verb (e.g., Optimized, Engineered, Spearheaded, Developed).
                    - Focus on achievements, technical systems managed, and industry standard duties.
                    - Do NOT include any introductory text or conclusions. Just give the raw bullet points starting with '• '.
                    """
                    ai_response = model.generate_content(ai_prompt)
                    st.session_state.ai_bullet_points = ai_response.text
                    st.success("Bullet points generated successfully!")

        st.subheader("🎓 Education")
        degree = st.text_input("Degree")
        college = st.text_input("College Name")
        grad_year = st.text_input("Graduation Year")

    st.markdown("---")
    st.subheader("👀 Live Resume Preview")
    
    # Renders the live view on the page using the state-saved AI bullets
    resume_markdown = f"""
    # {full_name if full_name else 'Your Name'}
    📧 {email} | 📱 {phone} | 🔗 {linkedin}
    ---
    ### 💼 Experience
    **{job_title}** at *{company}* ({duration})  
    {st.session_state.ai_bullet_points}
    
    ### 🎓 Education
    **{degree}** — *{college}* (Class of {grad_year})
    ### 🛠️ Skills
    `{skills}`
    """
    st.markdown(resume_markdown)

    if full_name:
        docx_data = generate_docx(
            full_name, email, phone, linkedin, skills, 
            job_title, company, duration, st.session_state.ai_bullet_points, 
            degree, college, grad_year
        )
        st.download_button(
            label="📥 Download Resume (.docx)",
            data=docx_data,
            file_name=f"{full_name.replace(' ', '_')}_Resume.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by a student")
