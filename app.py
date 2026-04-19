import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(page_title="HireXpert", layout="wide")

# Professional Job Categories with Skill Sets
# For a school project, this "Global Library" represents major industries
GLOBAL_JOB_LIBRARY = {
    "Healthcare": ["Doctor", "Nurse", "Pharmacist", "Surgeon", "Dentist", "Physiotherapist"],
    "Finance": ["Bank Manager", "Accountant", "Investment Analyst", "Loan Officer", "Auditor"],
    "Technology": ["Software Engineer", "Data Scientist", "Cybersecurity Analyst", "Web Developer"],
    "Education": ["Teacher", "Professor", "School Counselor", "Principal", "Librarian"],
    "Legal": ["Lawyer", "Judge", "Paralegal", "Legal Assistant", "Corporate Counsel"]
}

def extract_text(file):
    pdf = PdfReader(file)
    return " ".join([p.extract_text() for p in pdf.pages])

def get_similarity(resume, job_data):
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform([resume, job_data])
    return round(cosine_similarity(matrix)[0][1] * 100, 2)

st.title("HireXpert: AI Resume Scanner 🤖")

uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if st.button("Scan Against All Industries"):
    if uploaded_resume:
        text = extract_text(uploaded_resume)
        st.subheader("Industry Match Results:")
        
        results = []
        for industry, roles in GLOBAL_JOB_LIBRARY.items():
            # Compare resume to the combined keywords of that entire industry
            industry_keywords = " ".join(roles)
            score = get_similarity(text, industry_keywords)
            results.append((industry, score, roles))
        
        # Sort by best fit
        results.sort(key=lambda x: x[1], reverse=True)
        
        best_industry, best_score, best_roles = results[0]
        
        st.success(f"Top Industry Match: **{best_industry}** ({best_score}%)")
        st.write(f"**Recommended Roles for you:** {', '.join(best_roles)}")
        
        # Display other matches
        with st.expander("See Other Industry Scores"):
            for ind, sco, _ in results[1:]:
                st.write(f"{ind}: {sco}%")
    else:
        st.error("Please upload a resume first.")
