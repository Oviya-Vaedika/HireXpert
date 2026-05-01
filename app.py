import streamlit as st
import PyPDF2
import docx
import re
import google.generativeai as genai

# 1. Page Configuration & AI Setup
st.set_page_config(page_title="HireXpert", page_icon="🤖", layout="wide")

# --- SAFE API SETUP ---
try:
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"API Key not found or Config Error: {e}")

# 2. Support Functions
def extract_text(uploaded_file):
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

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return " ".join(text.split())

def get_ai_analysis(resume_text, jd_text):
    prompt = f"""
    Act as an expert HR Manager and ATS system. 
    Analyze the following Resume against the Job Description (JD).
    
    1. Start with a 'Match Score: [0-100]%' at the very top.
    2. List 'Top 5 Missing Keywords' that would improve the score.
    3. Provide a brief 'Improvement Summary'.
    
    JD: {jd_text}
    Resume: {resume_text}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {e}"

# 3. Custom Header
st.markdown("<h1>HireXpert 🤖 <span style='font-size: 0.5em; color: gray;'>Global AI Resume Screening</span></h1>", unsafe_allow_html=True)
st.divider()

# 4. Input Section
jd = st.text_area("Step 1: Paste the Job Description (JD):", placeholder="Paste the full job requirements here...", height=150) 
uploaded_file = st.file_uploader("Step 2: Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# 5. Logic & Output
if uploaded_file and jd:
    resume_raw = extract_text(uploaded_file)
    if resume_raw:
        resume_text_clean = clean_text(resume_raw)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Run AI Analysis", use_container_width=True):
                with st.spinner("AI is evaluating your resume..."):
                    result = get_ai_analysis(resume_text_clean, jd)
                    st.session_state['ai_report'] = result 
                    
                    st.subheader("AI Analysis Results")
                    st.markdown(result)

        with col2:
            if st.button("✨ Resume Tips", use_container_width=True):
                st.subheader("Quick Enhancements")
                st.write("#### 📝 Recommended Word Swaps:")
                weak_words = {"managed": "Spearheaded", "helped": "Facilitated", "led": "Orchestrated", "worked": "Collaborated"}
                found_any = False
                for weak, strong in weak_words.items():
                    if weak in resume_text_clean:
                        st.write(f"- Replace **'{weak}'** with: **'{strong}'**")
                        found_any = True
                if not found_any:
                    st.success("Your choice of action verbs looks strong!")

        if 'ai_report' in st.session_state:
            st.divider()
            st.download_button(
                label="📥 Download Full AI Report",
                data=st.session_state['ai_report'],
                file_name="hirexpert_ai_report.txt",
                mime="text/plain"
            )
    else:
        st.error("Could not extract text from the file.")
else:
    st.info("Upload your resume and enter a JD to start.")

st.divider()
st.caption("Made with ❤️ by a student developer")
