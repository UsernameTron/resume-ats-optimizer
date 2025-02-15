from app.core.enhanced_analyzer import EnhancedAnalyzer
from app.core.data_manager import DataManager
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # Initialize components
    data_manager = DataManager()
    analyzer = EnhancedAnalyzer(data_manager)
    
    # Read test files
    with open('test_resume.txt', 'r') as f:
        resume_text = f.read()
    with open('test_job.txt', 'r') as f:
        job_text = f.read()
        
    # Analyze resume
    result = analyzer.analyze_resume(resume_text, job_text)
    
    # Print results
    print("\nAnalysis Results:")
    print(f"Experience Match Score: {result.experience_match_score}")
    print(f"Overall ATS Score: {result.overall_score}")
    print(f"Skill Matches: {result.skill_matches}")
    
if __name__ == "__main__":
    main()
