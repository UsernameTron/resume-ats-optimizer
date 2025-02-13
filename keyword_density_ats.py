import os
import re
import glob
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
        """Train vectorizer using comprehensive job description data from CSV files"""
        training_texts = []
        skill_importance = {}
        
        # Load and process all CSV files
        csv_files = glob.glob("Generated_Job_Descriptions*.csv")
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                for _, row in df.iterrows():
                    # Combine relevant fields for comprehensive text analysis
                    text_components = []
                    
                    if 'Description' in df.columns:
                        text_components.append(str(row['Description']))
                    if 'Key Responsibilities' in df.columns:
                        text_components.append(str(row['Key Responsibilities']))
                    if 'Skills Required' in df.columns:
                        text_components.append(str(row['Skills Required']))
                        # Track skill importance
                        skills = str(row['Skills Required']).split(',')
                        for skill in skills:
                            skill = skill.strip()
                            skill_importance[skill] = skill_importance.get(skill, 0) + 1
                    
                    combined_text = ' '.join(text_components)
                    training_texts.append(combined_text)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
        
        # If no training texts were loaded, fall back to defaults
        if not training_texts:
            training_texts = [
                "customer experience leadership skills",
                "technical project management data analysis",
                "marketing communication strategic planning"
            ]
        
        # Store skill importance scores for later use
        self.skill_importance = {
            k: v/len(training_texts) 
            for k, v in skill_importance.items()
        }
        
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
        """Calculate match score between resume and job description with skill importance weighting"""
        if not self.vectorizer:
            return 0.0, set(), set()
            
        job_text = self.preprocess_text(job_desc)
        resume_text = self.preprocess_text(resume)
        
        # Calculate base similarity score
        vectors = self.vectorizer.transform([job_text, resume_text])
        base_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        # Extract terms
        terms = self.vectorizer.get_feature_names_out()
        job_terms = set(terms[vectors[0].nonzero()[1]])
        resume_terms = set(terms[vectors[1].nonzero()[1]])
        
        # Find matching and missing terms
        matches = job_terms & resume_terms
        missing = job_terms - resume_terms
        
        # Apply skill importance weighting
        importance_score = 0
        total_importance = 0
        
        for term in job_terms:
            weight = self.skill_importance.get(term, 0.1)  # Default weight for non-critical skills
            total_importance += weight
            if term in matches:
                importance_score += weight
                
        # Combine base similarity with importance-weighted score
        if total_importance > 0:
            weighted_score = importance_score / total_importance
            similarity = (0.7 * base_similarity) + (0.3 * weighted_score)
        else:
            similarity = base_similarity
        
        return round(similarity * 100, 2), matches, missing

    def _format_word_friendly_text(self, text):
        """Format text to be Word-friendly"""
        # Replace any special quotes or dashes with standard ASCII
        text = text.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u2014', '-').replace('\u2013', '-')
        # Ensure proper line breaks
        text = text.replace('\n', '\r\n')
        return text

    def optimize_resume(self, resume, job_desc):
        match_score, matches, missing = self.calculate_match(job_desc, resume)
        
        # Format missing keywords for better readability
        missing_keywords_formatted = '\r\n'.join([f'• {keyword}' for keyword in sorted(missing)])
        
        try:
            # Prepare system message with formatting instructions
            system_message = """You are an expert resume optimizer. Provide recommendations in this format:

1. SUMMARY OF ANALYSIS
- Brief overview of the match and key areas for improvement

2. KEYWORD INCORPORATION SUGGESTIONS
- Specific ways to naturally add missing keywords

3. CONTENT OPTIMIZATION RECOMMENDATIONS
- Bullet-point suggestions for improving content
- Focus on quantifiable achievements"""

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Job Description:\r\n{job_desc}\r\n\r\nCurrent Resume:\r\n{resume}\r\n\r\nMatch Score: {match_score:.1%}\r\n\r\nMissing Keywords:\r\n{missing_keywords_formatted}"}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            recommendations = self._format_word_friendly_text(response.choices[0].message.content)
            
            # Create a formatted output that's easy to copy-paste
            output = f"""RESUME OPTIMIZATION REPORT
{'='*50}

MATCH SCORE: {match_score:.1%}

KEY MISSING KEYWORDS:
{missing_keywords_formatted}

DETAILED RECOMMENDATIONS:
{recommendations}

{'='*50}
End of Report"""
            
            return output
            
        except Exception as e:
            return f"Optimization error: {str(e)}"

def main():
    st.set_page_config(page_title="Resume Optimizer", layout="wide")
    st.title("AI Resume Optimizer")
    
    # Add description
    st.markdown("""
    This AI-powered tool helps optimize your resume for ATS (Applicant Tracking Systems) by:
    - Analyzing keyword matches with job descriptions
    - Providing detailed recommendations for improvement
    - Suggesting natural ways to incorporate missing keywords
    """)
    
    optimizer = ResumeOptimizer()
    
    col1, col2 = st.columns(2)
    
    with col1:
        resume = st.text_area("Paste your resume here:", height=400, help="Paste the full text of your current resume")
    with col2:
        job_desc = st.text_area("Paste the job description here:", height=400, help="Paste the complete job description you're applying for")
        
    if st.button("Analyze and Optimize Resume"):
        if not resume.strip() or not job_desc.strip():
            st.warning("Please provide both your resume and the job description.")
            return
        
        with st.spinner("Analyzing your resume and generating recommendations..."):
            # Calculate initial match
            initial_score, matches, missing = optimizer.calculate_match(job_desc, resume)
            
            # Get optimization recommendations
            optimization_report = optimizer.optimize_resume(resume, job_desc)
            
            # Display results
            st.subheader("Analysis Results")
            
            # Show match score
            st.metric("ATS Match Score", f"{initial_score}%", help="Higher scores indicate better alignment with the job description")
            
            # Display the full optimization report in a clean format
            st.markdown("### Detailed Optimization Report")
            st.text_area(
                "Copy the full report below:",
                optimization_report,
                height=400
            )
            
            # Add download button for the report
            st.download_button(
                label="Download Report as Text",
                data=optimization_report,
                file_name="resume_optimization_report.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
