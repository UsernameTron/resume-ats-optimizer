import streamlit as st
import logging
import pandas as pd
import time
from pathlib import Path
from typing import List, Optional
from app.core.data_manager import DataManager
from app.core.enhanced_analyzer import EnhancedAnalyzer
from app.core.resource_monitor import ResourceMonitor
from app.utils.nltk_utils import ensure_nltk_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
@st.cache_resource
def init_components():
    data_manager = DataManager()
    analyzer = EnhancedAnalyzer(data_manager)
    monitor = ResourceMonitor()
    return data_manager, analyzer, monitor

def main():
    st.set_page_config(
        page_title="Enhanced ATS Resume Optimizer",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Enhanced ATS Resume Optimizer")
    st.markdown("""
    Optimize your resume for Applicant Tracking Systems (ATS) with advanced AI-powered analysis.
    Get detailed feedback, keyword optimization, and industry-specific recommendations.
    """)
    
    # Initialize components
    data_manager, analyzer, monitor = init_components()
    
    # Create two columns for resume and job description
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Resume")
        upload_mode = st.radio(
            "Choose input method",
            ["Paste Resume", "Upload File", "Batch Process"]
        )
        
        if upload_mode == "Paste Resume":
            resume_text = st.text_area(
                "Paste your resume here",
                height=300,
                placeholder="Paste your resume content here..."
            )
            resumes = [resume_text] if resume_text else []
        elif upload_mode == "Upload File":
            uploaded_file = st.file_uploader("Upload your resume", type=['txt', 'pdf', 'docx'])
            if uploaded_file:
                # TODO: Add file processing based on type
                resume_text = uploaded_file.read().decode()
                resumes = [resume_text]
            else:
                resumes = []
        else:  # Batch Process
            uploaded_files = st.file_uploader(
                "Upload multiple resumes",
                type=['txt', 'pdf', 'docx'],
                accept_multiple_files=True
            )
            resumes = []
            if uploaded_files:
                for file in uploaded_files:
                    # TODO: Add file processing based on type
                    resume_text = file.read().decode()
                    resumes.append(resume_text)
        
    with col2:
        st.subheader("Job Description")
        job_description = st.text_area(
            "Paste the job description",
            height=300,
            placeholder="Paste the job description here..."
        )
        
    # Additional options
    with st.expander("Advanced Options"):
        target_industry = st.selectbox(
            "Target Industry",
            ["Technology", "Finance", "Healthcare", "Marketing", "Other"],
            index=0
        )
        
        optimize_keywords = st.checkbox("Optimize Keywords", value=True)
        restructure_content = st.checkbox("Suggest Content Restructuring", value=True)
    
    # Analysis button
    if st.button("Analyze and Optimize Resume", type="primary"):
        if not resume_text or not job_description:
            st.error("Please provide both resume and job description.")
            return
            
        try:
            with st.spinner("Analyzing your resume..."):
                # Get performance metrics
                metrics = monitor.get_performance_metrics()
                
                # Track start time
                start_time = time.time()
                
                # Analyze resumes
                if len(resumes) > 1:
                    results = analyzer.analyze_batch(
                        resumes=resumes,
                        job_description=job_description,
                        target_industry=target_industry if target_industry != "Other" else None
                    )
                    # Show batch results summary
                    st.subheader(f"Batch Analysis Results ({len(results)} resumes)")
                    summary_df = pd.DataFrame([
                        {
                            'ATS Score': r.ats_score,
                            'Industry Match': r.industry_match_score,
                            'Experience Match': r.experience_match_score,
                            'Overall Score': r.overall_score,
                            'Error': r.error if hasattr(r, 'error') else None
                        } for r in results
                    ])
                    st.dataframe(summary_df)
                    
                    # Allow user to select specific resume to view details
                    selected_idx = st.selectbox(
                        "Select resume to view details",
                        range(len(results)),
                        format_func=lambda x: f"Resume {x+1}"
                    )
                    result = results[selected_idx]
                else:
                    result = analyzer.analyze_resume(
                        resume_text=resumes[0],
                        job_description=job_description,
                        target_industry=target_industry if target_industry != "Other" else None
                    )
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Display results
                st.success("Analysis complete!")
                
                # Create tabs for different sections
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Score Overview",
                    "🎯 Skill Analysis",
                    "📝 Recommendations",
                    "⚙️ System Metrics"
                ])
                
                with tab1:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("ATS Score", f"{result.ats_score:.0%}")
                    with col2:
                        st.metric("Industry Match", f"{result.industry_match_score:.0%}")
                    with col3:
                        st.metric("Experience Match", f"{result.experience_match_score:.0%}")
                    with col4:
                        st.metric("Overall Score", f"{result.overall_score:.0%}")
                        
                with tab2:
                    # Skill matches
                    st.subheader("Skill Matches")
                    for skill, score in result.skill_matches.items():
                        st.progress(score, text=f"{skill}: {score:.0%}")
                    
                    # Missing skills
                    if result.missing_critical_skills:
                        st.error("Missing Critical Skills")
                        st.write(", ".join(result.missing_critical_skills))
                        
                with tab3:
                    st.subheader("Improvement Suggestions")
                    for i, suggestion in enumerate(result.improvement_suggestions, 1):
                        st.info(f"{i}. {suggestion}")
                    
                    if optimize_keywords:
                        st.subheader("Optimized Resume")
                        st.text_area(
                            "Copy this optimized version",
                            value=result.optimized_resume,
                            height=200
                        )
                        
                with tab4:
                    st.subheader("System Performance")
                    
                    # Display memory usage
                    memory_usage = monitor.get_memory_usage()
                    st.progress(
                        memory_usage / 100,
                        text=f"Memory Usage: {memory_usage:.1f}%"
                    )
                    
                    # Display processing metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Processing Time",
                            f"{processing_time:.2f}s"
                        )
                    with col2:
                        st.metric(
                            "Resumes Processed",
                            len(resumes)
                        )
                    
                    # Display detailed metrics
                    metrics_df = pd.DataFrame({
                        "Metric": metrics.keys(),
                        "Value": metrics.values()
                    })
                    st.dataframe(metrics_df)
                    
                    # Display any warnings or errors
                    if hasattr(result, 'error') and result.error:
                        st.error(f"Error during analysis: {result.error}")
                    
                # Download button for report
                report = generate_report(result)
                st.download_button(
                    label="Download Analysis Report",
                    data=report,
                    file_name="ats_analysis_report.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error during analysis: {error_msg}")
            st.error(f"An error occurred during analysis: {error_msg}")
            monitor.log_error("resume_analysis", error_msg)

def generate_report(result) -> str:
    """Generate a detailed report from analysis results"""
    report = []
    report.append("ATS RESUME ANALYSIS REPORT")
    report.append("=" * 50)
    report.append(f"\nOVERALL SCORES")
    report.append(f"ATS Score: {result.ats_score:.0%}")
    report.append(f"Industry Match: {result.industry_match_score:.0%}")
    report.append(f"Experience Match: {result.experience_match_score:.0%}")
    report.append(f"Overall Score: {result.overall_score:.0%}")
    
    report.append("\nSKILL ANALYSIS")
    report.append("-" * 20)
    for skill, score in result.skill_matches.items():
        report.append(f"{skill}: {score:.0%}")
    
    if result.missing_critical_skills:
        report.append("\nMISSING CRITICAL SKILLS")
        report.append("-" * 20)
        report.append(", ".join(result.missing_critical_skills))
    
    report.append("\nRECOMMENDATIONS")
    report.append("-" * 20)
    for i, suggestion in enumerate(result.improvement_suggestions, 1):
        report.append(f"{i}. {suggestion}")
    
    return "\n".join(report)

if __name__ == "__main__":
    main()
