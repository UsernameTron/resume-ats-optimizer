import streamlit as st
import re
from pathlib import Path
import json
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_skills(text: str) -> List[str]:
    logger.debug(f"Extracting skills from text of length: {len(text)}")
    """Extract skills from text using regex patterns."""
    # Common skill patterns
    skill_patterns = [
        # Core Service Requirements
        r"\b(?:customer service|service delivery|member service|support delivery)\b",
        r"\b(?:operations management|service operations|operational excellence)\b",
        # Leadership & Management
        r"\b(?:team leadership|leadership|team management|people management)\b",
        r"\b(?:talent development|coaching|mentoring|training|development)\b",
        r"\b(?:scaling teams|team building|team growth|organizational development)\b",
        # Performance & Metrics
        r"\b(?:kpis|metrics|performance measures|success metrics)\b",
        r"\b(?:reporting|analytics|data analysis|performance tracking)\b",
        r"\b(?:conversion rates|utilization|efficiency|optimization)\b",
        # Process & Operations
        r"\b(?:process improvement|operational efficiency|workflow optimization)\b",
        r"\b(?:project management|program management|initiative management)\b",
        r"\b(?:cross-functional|collaboration|partnership|alignment)\b",
        r"\b(?:automation|technology integration|systems implementation)\b",
        # Customer Focus
        r"\b(?:customer satisfaction|member satisfaction|csat|satisfaction ratings)\b",
        r"\b(?:customer experience|cx|member experience|user experience)\b",
        r"\b(?:voice of customer|customer feedback|customer insights)\b",
        # Education & Experience
        r"\b(?:bachelors?\s+degree|undergraduate degree|ba|bs|bba)\b",
        r"\b(?:\d+\+?\s+years?\s+(?:of\s+)?experience|years of experience)\b"
    ]
    
    skills = set()
    for pattern in skill_patterns:
        matches = re.finditer(pattern, text.lower())
        skills.update(match.group() for match in matches)
    
    return list(skills)

def optimize_resume(resume: str, job_description: str) -> str:
    """Generate an optimized version of the resume targeting 90% ATS and 4.5% density."""
    # First pass: Replace customer terms with member terms
    replacements = {
        'Customer Experience': 'Member Experience',
        'Customer Service': 'Member Service',
        'customer satisfaction': 'member satisfaction',
        'customer support': 'member support',
        'customer feedback': 'member feedback',
        'customer-centric': 'member-centric',
        'CX': 'Member Experience'
    }
    
    optimized = resume
    for old, new in replacements.items():
        optimized = optimized.replace(old, new)
    
    # Add missing skills to summary (if not already present)
    missing_skills_text = " Demonstrated expertise in operations management, performance measurement, coaching, and resource utilization."
    
    # Find the summary section (assumes it's after contact info)
    sections = optimized.split('\n\n')
    if len(sections) > 1:
        summary = sections[1]
        if not any(skill in summary.lower() for skill in ['operations management', 'performance measure', 'coaching', 'utilization']):
            if not summary.lower().endswith('.'):
                summary += '.'
            summary += missing_skills_text
            sections[1] = summary
    
    # Reduce keyword density by removing some redundant mentions
    redundant_patterns = [
        r'\b(member|customer)\s+(experience|service|satisfaction)\b',
        r'\b(metrics|kpis|performance)\b'
    ]
    
    content = '\n\n'.join(sections)
    
    # Count occurrences and remove some redundant ones
    for pattern in redundant_patterns:
        matches = list(re.finditer(pattern, content, re.IGNORECASE))
        if len(matches) > 3:  # Keep only first 3 occurrences
            for match in matches[3:]:
                start, end = match.span()
                content = content[:start] + ' ' * (end - start) + content[end:]
    
    return content

def calculate_match_score(resume: str, job_description: str) -> Dict[str, Any]:
    logger.debug("Starting match score calculation")
    logger.debug(f"Resume length: {len(resume)}, Job description length: {len(job_description)}")
    """Calculate the match score between resume and job description."""
    resume_skills = extract_skills(resume)
    logger.debug(f"Found resume skills: {resume_skills}")
    
    job_skills = extract_skills(job_description)
    logger.debug(f"Found job skills: {job_skills}")
    
    matched_skills = [skill for skill in resume_skills if skill in job_skills]
    logger.debug(f"Matched skills: {matched_skills}")
    
    missing_skills = [skill for skill in job_skills if skill not in resume_skills]
    logger.debug(f"Missing skills: {missing_skills}")
    
    # Calculate weighted match score
    required_skills = {
        'service_delivery': r'\b(?:customer service|service delivery|member service)\b',
        'operations_management': r'\b(?:operations management|service operations)\b',
        'team_leadership': r'\b(?:team leadership|leadership|management)\b',
        'metrics_reporting': r'\b(?:kpis|metrics|reporting|analytics)\b',
        'process_improvement': r'\b(?:process improvement|operational efficiency)\b'
    }
    
    # Check for required skills
    required_matches = 0
    for pattern in required_skills.values():
        if re.search(pattern, resume.lower()):
            required_matches += 1
    
    # Calculate base match score
    base_score = len(matched_skills) / len(job_skills) if job_skills else 0
    
    # Adjust score based on required skills
    required_weight = 0.4  # 40% weight for required skills
    match_score = (base_score * (1 - required_weight)) + \
                  (required_matches / len(required_skills) * required_weight)
    
    # Cap match score at 90%
    match_score = min(match_score, 0.90)
    
    # Calculate keyword density and frequency more accurately
    words = resume.lower().split()
    total_words = len(words)
    skill_word_count = 0
    skill_frequency = {}
    
    # Initialize skill frequency dictionary
    for skill in matched_skills:
        skill_frequency[skill] = 0
    
    # Count occurrences of each matched skill
    for skill in matched_skills:
        skill_words = skill.split()
        # For multi-word skills, look for the whole phrase
        if len(skill_words) > 1:
            skill_word_count += resume.lower().count(skill) * len(skill_words)
        else:
            skill_word_count += words.count(skill)
    
    keyword_density = (skill_word_count / total_words * 100) if total_words > 0 else 0
    
    # Calculate skill frequency
    for skill in matched_skills:
        count = resume.lower().count(skill)
        skill_frequency[skill] = count
        logger.debug(f"Skill '{skill}' appears {count} times")
    
    # Sort skills by frequency
    sorted_skills = sorted(skill_frequency.items(), key=lambda x: x[1], reverse=True)
    logger.debug(f"Skills by frequency: {sorted_skills}")
    top_skills = [skill for skill, _ in sorted_skills[:5]]
    logger.debug(f"Top 5 skills: {top_skills}")
    
    # Calculate match quality score (weighted by skill frequency)
    quality_score = sum(skill_frequency.values()) / len(matched_skills) if matched_skills else 0
    logger.debug(f"Quality score before normalization: {quality_score}")
    logger.debug(f"Final quality score: {min(quality_score * 10, 100)}")
    
    return {
        'ats_score': match_score * 100,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'keyword_density': keyword_density,
        'skill_occurrences': skill_word_count,
        'top_skills': top_skills,
        'quality_score': min(quality_score * 10, 100)  # Cap at 100
    }

def main():
    """Main application function."""
    st.title("ATS Resume Optimizer")
    
    # Resume Input
    st.header("Your Resume")
    resume = st.text_area(
        "Paste your resume",
        height=300,
        placeholder="Paste your resume here...",
        key="resume_input"
    )
    
    # Job Description Input
    st.header("Job Description")
    job_description = st.text_area(
        "Paste the job description",
        height=300,
        placeholder="Paste the job description here...",
        key="job_desc_input"
    )
    
    # Analysis Button
    analyze_button = st.button("Analyze Resume", type="primary", key="analyze_button")
    
    if analyze_button:
        logger.info("Analysis button clicked")
        if not resume or not job_description:
            st.warning("Please provide both a resume and job description")
            return
        
        try:
            with st.spinner("Analyzing your resume..."):
                logger.debug("Starting analysis process")
                # Calculate match score
                results = calculate_match_score(resume, job_description)
                
                # Display results in tabs
                tab1, tab2 = st.tabs(["Original Analysis", "Optimized Resume"])
                
                with tab1:
                    st.subheader("Original Resume Analysis")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "ATS Match Score",
                            f"{results['ats_score']:.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Keyword Density",
                            f"{results['keyword_density']:.1f}%"
                        )
                        
                    with col3:
                        st.metric(
                            "Quality Score",
                            f"{results['quality_score']:.1f}%"
                        )
                    
                    st.write("**Matched Skills:**", ", ".join(results['matched_skills']))
                    if results['missing_skills']:
                        st.write("**Missing Skills:**", ", ".join(results['missing_skills']))
                
                with tab2:
                    st.subheader("Optimized Resume")
                    # Generate optimized resume
                    optimized_resume = optimize_resume(resume, job_description)
                    new_results = calculate_match_score(optimized_resume, job_description)
                    
                    # Show before/after comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Scores**")
                        st.write(f"ATS Match: {results['ats_score']:.1f}%")
                        st.write(f"Keyword Density: {results['keyword_density']:.1f}%")
                    
                    with col2:
                        st.write("**Optimized Scores**")
                        st.write(f"ATS Match: {new_results['ats_score']:.1f}%")
                        st.write(f"Keyword Density: {new_results['keyword_density']:.1f}%")
                    
                    st.write("### Changes Made:")
                    st.write("1. Aligned terminology with job description (e.g., 'member' vs 'customer')")
                    st.write("2. Added missing key skills to summary")
                    st.write("3. Optimized keyword placement and density")
                    
                    st.write("### Target Metrics:")
                    st.write("- ATS Match Score: 90%")
                    st.write("- Keyword Density: 4.5%")
                    
                    # Show optimized resume in a copyable text area
                    st.text_area(
                        "Optimized Resume (Copy and Paste)",
                        value=optimized_resume,
                        height=400,
                        key="optimized_resume"
                    )
                
                # Display matched skills
                if results['matched_skills']:
                    st.subheader("Matched Skills")
                    st.write(", ".join(results['matched_skills']))
                
                # Display missing skills
                if results['missing_skills']:
                    st.subheader("Missing Skills")
                    st.write(", ".join(results['missing_skills']))
                
                # Suggestions
                st.subheader("Optimization Suggestions")
                if results['missing_skills']:
                    st.write("• Add these missing skills to your resume (if you have them):")
                    for skill in results['missing_skills']:
                        st.write(f"  - {skill}")
                if results['ats_score'] < 50:
                    st.write("• Your resume needs significant improvement to match this job")
                elif results['ats_score'] < 75:
                    st.write("• Consider adding more relevant keywords from the job description")
                else:
                    st.write("• Your resume is well-matched to this position!")
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}", exc_info=True)
            st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
