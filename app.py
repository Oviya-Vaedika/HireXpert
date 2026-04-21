import streamlit as st
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# 1. Page Configuration & Branding
st.set_page_config(page_title="HireXpert", page_icon="🌍", layout="centered")

# 2. File Processing Functions
def extract_text(uploaded_file):
    """Extracts text from PDF, DOCX, or TXT."""
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

def is_resume(text):
    """Basic check to see if the file is a resume."""
    keywords = ['experience', 'education', 'skills', 'projects', 'summary', 'contact', 'employment']
    count = sum(1 for word in keywords if word in text.lower())
    return count >= 2

# 3. Industry & Scoring Logic
def identify_global_category(resume_text):
    """Categorizes resume into major global industries if match is low."""
    industries = {
        "Healthcare": "patient medicine clinical hospital nursing health surgery doctor",
        "Education": "curriculum student school teaching lesson pedagogy classroom professor",
        "Skilled Trades": "electrical plumbing repair maintenance tools construction mechanical",
        "Hospitality/Tourism": "hotel travel guest service tourism restaurant culinary chef",
        "Finance/Accounting": "budget audit tax accounting banking financial investment",
        "Creative Arts/Design": "graphic design creative illustration video edit content artist",
        "Sales/Marketing": "lead generation seo marketing sales campaign customer market",
        "Logistics/Warehouse": "inventory shipping supply chain warehouse logistics forklift",
        "Technology/IT": "software developer engineer python java javascript cloud cybersecurity"
    }
    
    best_match = "General Professional"
    highest_sim = 0
    
    for industry, keywords in industries.items():
        docs = [resume_text.lower(), keywords]
        vec = TfidfVectorizer().fit_transform(docs)
        sim = cosine_similarity(vec[0:1], vec[1:2])
        if sim > highest_sim:
            highest_sim = sim
            best_match = industry
    return best_match

# 4. UI Layout
st.title("HireXpert 🌍")
st.markdown("### Global AI Resume Screening & Optimization")

jd = st.text_area("Step 1: Paste the Job Description (JD):", placeholder="Paste any job requirements from any industry...", height=150)
uploaded_file = st.file_uploader("Step 2: Upload Resume", type=["pdf", "docx", "txt"])

if uploaded_file and jd:
    resume_content = extract_text(uploaded_file)
    
    if not is_resume(resume_content):
        st.error("🚨 The uploaded file does not appear to be a professional resume. Please check and re-upload.")
    else:
        # Create two action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Analyze Resume", use_container_width=True):
                st.divider()
                # Calculate ATS Score
                docs = [jd.lower(), resume_content.lower()]
                vectorizer = TfidfVectorizer().fit_transform(docs)
                score = round(cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0] * 100, 2)
                
                st.metric("ATS Match Score", f"{score}%")
                
                if score >= 65:
                    st.success("✅ **Result: Eligible.** You have a strong keyword match for this role.")
                else:
                    st.error("❌ **Result: Not Eligible.** This specific role doesn't match your current resume.")
                    industry_type = identify_global_category(resume_content)
                    st.info(f"💡 **HireXpert Suggestion:** Your skills align best with the **{industry_type}** sector. Try searching for roles in that field!")

        with col2:
            if st.button("✨ Improve Resume", use_container_width=True):
                st.divider()
                st.subheader("Actionable Improvements")
                
                # Identify Skill Gaps
                jd_words = set(re.findall(r'\w+', jd.lower()))
                resume_words = set(re.findall(r'\w+', resume_content.lower()))
                # Filter out short/common words and find what's missing
                missing = [w for w in (jd_words - resume_words) if len(w) > 4]
                
                st.markdown("#### 🛠 Keywords You Should Add:")
                if missing:
                    st.write(", ".join(missing[:15]))
                else:
                    st.write("Your resume already contains the key terms from this JD!")
                
                st.markdown("#### 💡 Pro Tips for a Better Score:")
                st.write("- **Use Action Verbs:** Instead of 'I did', use 'Spearheaded' or 'Implemented'.")
                st.write("- **Add Numbers:** Quantify your work (e.g., 'Managed a team of 5' or 'Cut costs by 15%').")
                st.write("- **Simple Layout:** Avoid tables and graphics; ATS scanners read plain text best.")

else:
    st.info("Please enter a Job Description and upload a Resume to get started.")

# Footer
st.markdown("---")
st.caption("Powered by HireXpert - Your Global Career Ally")
