import json
from pathlib import Path
import re
from collections import Counter

def analyze_resume(resume_text, job_description):
    # Key metrics to look for
    key_metrics = {
        'nps': 'Net Promoter Score',
        'csat': 'Customer Satisfaction',
        'fcr': 'First-Call Resolution',
        'retention': 'Customer Retention',
        'efficiency': 'Operational Efficiency',
        'cost': 'Cost Reduction',
        'revenue': 'Revenue Impact'
    }
    
    # Key tools and technologies
    key_tools = {
        'crm': ['salesforce', 'zendesk', 'totango', 'ringcentral'],
        'analytics': ['power bi', 'tableau', 'excel', 'sql', 'python'],
        'ai_ml': ['tensorflow', 'scikit-learn', 'machine learning', 'nlp', 'ai'],
        'project_mgmt': ['asana', 'trello', 'jira'],
        'communication': ['slack', 'teams', 'zoom']
    }
    
    # Calculate matches
    metrics_found = []
    for metric, desc in key_metrics.items():
        if re.search(rf'\b{metric}\b', resume_text.lower()) or re.search(rf'\b{desc}\b', resume_text.lower()):
            metrics_found.append(desc)
    
    tools_found = []
    for category, tools in key_tools.items():
        for tool in tools:
            if re.search(rf'\b{tool}\b', resume_text.lower()):
                tools_found.append(tool)
    
    # Extract quantitative achievements
    achievements = re.findall(r'\d+%|\$\d+(?:,\d+)*(?:\.\d+)?(?:\s*[Mm]illion|\s*[Kk])?|\d+\s*(?:point|pts?)', resume_text)
    
    # Analyze certifications
    certifications = re.findall(r'(?:Certification|Certificate|Certified)\s+(?:in\s+)?([^.\n]+)', resume_text)
    
    # Calculate keyword density
    words = re.findall(r'\b\w+\b', resume_text.lower())
    word_freq = Counter(words)
    total_words = len(words)
    
    key_terms = ['ai', 'ml', 'customer', 'experience', 'analytics', 'leadership', 'strategy']
    keyword_density = {word: (word_freq[word] / total_words) * 100 for word in key_terms}
    
    return {
        'metrics_found': metrics_found,
        'tools_found': tools_found,
        'achievements': achievements,
        'certifications': certifications,
        'keyword_density': keyword_density
    }

def main():
    # Load resume
    resume_file = Path.home() / 'CascadeProjects' / 'enhanced-ats-optimizer' / 'data' / 'stored_resume.json'
    with open(resume_file) as f:
        resume_data = json.load(f)
    resume_text = resume_data['resume_text']
    
    # Load job description
    with open('sample_job.txt') as f:
        job_description = f.read()
    
    # Analyze
    results = analyze_resume(resume_text, job_description)
    
    # Print results
    print("\n=== RESUME ANALYSIS RESULTS ===")
    
    print("\nKey Metrics Found:")
    for metric in results['metrics_found']:
        print(f"✓ {metric}")
    
    print("\nTools & Technologies:")
    for tool in results['tools_found']:
        print(f"✓ {tool}")
    
    print("\nQuantitative Achievements:")
    for achievement in results['achievements']:
        print(f"• {achievement}")
    
    print("\nCertifications:")
    for cert in results['certifications']:
        print(f"• {cert}")
    
    print("\nKeyword Density:")
    for term, density in results['keyword_density'].items():
        print(f"• {term}: {density:.1f}%")

if __name__ == "__main__":
    main()
