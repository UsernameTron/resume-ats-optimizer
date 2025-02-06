import os
import re
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class ResumeOptimizer:
    def __init__(self):
        self.vectorizer = self._train_vectorizer()

    def _train_vectorizer(self):
        training_texts = [
            "customer experience leadership skills",
            "technical project management data analysis",
            "marketing communication strategic planning"
        ]

        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        vectorizer.fit(training_texts)
        return vectorizer

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'(\d+)(\s*-\s*\d+|\+)', r'\1plus', text)
        text = re.sub(r'(\d+)%', r'\1percent', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())

    def calculate_match(self, job_desc, resume):
        if not self.vectorizer:
            return 0.0, set(), set()
            
        job_text = self.preprocess_text(job_desc)
        resume_text = self.preprocess_text(resume)
        
        vectors = self.vectorizer.transform([job_text, resume_text])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        terms = self.vectorizer.get_feature_names_out()
        job_terms = set(terms[vectors[0].nonzero()[1]])
        resume_terms = set(terms[vectors[1].nonzero()[1]])
        
        matches = job_terms & resume_terms
        missing = job_terms - resume_terms
        
        return round(similarity * 100, 2), matches, missing

    def optimize_resume(self, resume, job_desc):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert resume optimizer. Enhance the resume to match the job description while maintaining professional integrity."},
                    {"role": "user", "content": f"Job Description:\n{job_desc}\n\nResume:\n{resume}\n\nOptimize the resume to improve ATS match and highlight relevant skills and experiences."}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Optimization error: {str(e)}"

def main():
    st.set_page_config(page_title="Resume Optimizer", layout="wide")
    st.title("AI Resume Optimizer")
    
    optimizer = ResumeOptimizer()
    
    col1, col2 = st.columns(2)
    
    with col1:
        resume = st.text_area("Current Resume:", height=400)
    with col2:
        job_desc = st.text_area("Job Description:", height=400)
        
    if st.button("Optimize Resume"):
        if not resume.strip() or not job_desc.strip():
            st.warning("Please provide both resume and job description.")
            return
            
        # Calculate initial match
        initial_score, matches, missing = optimizer.calculate_match(job_desc, resume)
        
        # Optimize resume
        optimized_resume = optimizer.optimize_resume(resume, job_desc)
        
        # Calculate new match score
        new_score, new_matches, new_missing = optimizer.calculate_match(job_desc, optimized_resume)
        
        # Display results
        st.subheader("Resume Optimization Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Initial ATS Match", f"{initial_score}%")
        with col2:
            st.metric("Optimized ATS Match", f"{new_score}%")
        
        with st.expander("Original Resume"):
            st.text(resume)
        
        with st.expander("Optimized Resume"):
            st.text(optimized_resume)
        
        with st.expander("Keyword Analysis"):
            st.write("Initial Matching Keywords:")
            st.write(", ".join(sorted(list(matches))[:15]) or "No matching keywords")
            
            st.write("\nOptimized Matching Keywords:")
            st.write(", ".join(sorted(list(new_matches))[:15]) or "No matching keywords")
            
            if new_missing:
                st.write("\nRemaining Suggested Keywords:")
                st.write(", ".join(sorted(list(new_missing))[:10]))

if __name__ == "__main__":
    main()
