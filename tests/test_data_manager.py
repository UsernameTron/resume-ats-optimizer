import pytest
import pandas as pd
from pathlib import Path
from app.core.data_manager import DataManager, JobRequirements

@pytest.fixture
def test_data_dir(tmp_path):
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Create test CSV files with more comprehensive data
    df1 = pd.DataFrame({
        'Job Title': ['Software Engineer', 'Data Scientist', 'ML Engineer'],
        'Industry': ['Technology', 'Technology', 'Technology'],
        'Responsibilities': [
            'Develop web applications; Write clean code; Implement APIs',
            'Build ML models; Analyze data; Deploy models to production',
            'Design deep learning systems; Train models; Evaluate performance'
        ],
        'Qualifications': [
            'Python expertise; JavaScript and Node.js; Git experience',
            'Python, TensorFlow, and PyTorch; Statistics; Machine Learning',
            'Deep Learning frameworks; Computer Vision; NLP'
        ],
        'Location': ['San Francisco', 'New York', 'Seattle'],
        'Salary Range': ['$100,000 - $150,000', '$90,000 - $130,000', '$110,000 - $160,000'],
        'Job Type': ['Full-time', 'Full-time', 'Full-time']
    })
    
    df2 = pd.DataFrame({
        'Job Title': ['Frontend Developer', 'DevOps Engineer', 'Cloud Architect'],
        'Industry': ['Technology', 'Technology', 'Technology'],
        'Responsibilities': [
            'Build responsive UIs; Optimize performance; Write unit tests',
            'Manage infrastructure; Deploy applications; Monitor systems',
            'Design cloud architecture; Implement security; Optimize costs'
        ],
        'Qualifications': [
            'React.js and Vue.js; TypeScript; Modern CSS',
            'AWS and GCP; Docker; Kubernetes; Linux',
            'AWS Solutions Architect; Terraform; CloudFormation'
        ],
        'Location': ['Austin', 'Remote', 'Boston'],
        'Salary Range': ['$85,000 - $125,000', '$120,000 - $170,000', '$130,000 - $180,000'],
        'Job Type': ['Full-time', 'Full-time', 'Full-time']
    })
    
    # Save test data
    df1.to_csv(data_dir / "Generated_Job_Descriptions.csv", index=False)
    df2.to_csv(data_dir / "generated_job_descriptions (2).csv", index=False)
    
    return data_dir

@pytest.fixture
def data_manager(test_data_dir):
    return DataManager(str(test_data_dir))

def test_initialize_data(data_manager):
    """Test data initialization"""
    # We have 6 jobs in the test data
    assert len(data_manager.job_requirements) == 6
    
    # Check specific job titles from our test data
    assert 'ML Engineer' in data_manager.job_requirements
    assert 'Data Scientist' in data_manager.job_requirements
    assert 'Software Engineer' in data_manager.job_requirements
    assert 'Frontend Developer' in data_manager.job_requirements
    assert 'DevOps Engineer' in data_manager.job_requirements
    assert 'Cloud Architect' in data_manager.job_requirements
    
    # Verify some job details
    ml_job = data_manager.job_requirements['ML Engineer']
    assert 'Machine Learning' in ml_job.required_skills
    assert 'Deep Learning' in ml_job.required_skills
    assert ml_job.experience_years == 3

def test_data_integrity(data_manager):
    """Verify test data integrity"""
    # Check all required jobs are present
    required_jobs = ['ML Engineer', 'Data Scientist', 'Software Engineer', 
                    'Frontend Developer', 'DevOps Engineer', 'Cloud Architect']
    for job in required_jobs:
        assert job in data_manager.job_requirements, f"Missing job: {job}"

def test_extract_experience_years(data_manager):
    """Test experience years extraction"""
    dm = data_manager
    
    # Test explicit years
    assert dm._extract_experience_years("5+ years experience") == 5
    assert dm._extract_experience_years("3-5 years required") == 5
    
    # Test work history
    assert dm._extract_experience_years("2020 - present: Senior Developer") >= 3
    assert dm._extract_experience_years("2018 - 2023: Team Lead") == 5
    
    # Test no experience mentioned
    assert dm._extract_experience_years("") == 0
    assert dm._extract_experience_years("Great communication skills") == 0

def test_parse_salary_range(data_manager):
    """Test salary range parsing"""
    dm = data_manager
    
    # Test k notation
    assert dm._parse_salary_range("$70k - $100k") == (70000.0, 100000.0)
    
    # Test full numbers
    assert dm._parse_salary_range("$70,000 - $100,000") == (70000.0, 100000.0)
    
    # Test without currency symbol
    assert dm._parse_salary_range("70k - 100k") == (70000.0, 100000.0)
    
    # Test with "salary range" prefix
    assert dm._parse_salary_range("Salary Range: $70k - $100k") == (70000.0, 100000.0)
    
    # Test invalid formats
    assert dm._parse_salary_range("") == (0.0, 0.0)
    assert dm._parse_salary_range("Competitive") == (0.0, 0.0)

def test_extract_skills(data_manager):
    """Test skill extraction"""
    dm = data_manager
    
    text = """
    Required Skills:
    - Python programming
    - JavaScript and Node.js
    - React.js and Vue.js
    - SQL databases (MySQL, PostgreSQL)
    - Machine Learning and TensorFlow
    - C++ for performance optimization
    """
    
    skills = dm._extract_skills(text)
    
    # Test technical terms
    assert "python" in skills
    assert "javascript" in skills
    assert "node.js" in skills
    assert "react.js" in skills
    assert "vue.js" in skills
    assert "sql" in skills
    assert "mysql" in skills
    assert "postgresql" in skills
    assert "machine learning" in skills
    assert "tensorflow" in skills
    assert "c++" in skills

def test_match_skills(data_manager):
    """Test skill matching"""
    dm = data_manager
    
    # Create a job requirement
    job_req = JobRequirements(
        title="Software Engineer",
        industry="Technology",
        responsibilities=["Develop software"],
        qualifications=["Python expertise"],
        required_skills={"python", "javascript", "sql"},
        experience_years=5,
        location="Remote",
        job_type="Full-Time",
        salary_range=(100000, 150000)
    )
    
    # Test exact matches
    resume = "Expert in Python and JavaScript with SQL database experience"
    matches = dm.match_skills(resume, job_req)
    
    assert matches["python"] > 0.7  # High confidence for exact match
    assert matches["javascript"] > 0.7
    assert matches["sql"] > 0.7
    
    # Test partial matches
    resume = "Experience with programming and web development"
    matches = dm.match_skills(resume, job_req)
    
    assert all(0 <= score <= 1 for score in matches.values())  # Scores should be normalized

def test_industry_weights(data_manager):
    """Test industry weight calculation"""
    dm = data_manager
    
    # Create job requirements with specific skills
    dm.job_requirements = {
        "test": JobRequirements(
            title="Software Engineer",
            industry="Technology",
            responsibilities=["Develop software using Python and JavaScript"],
            qualifications=["5+ years experience in web development"],
            required_skills={"python", "javascript", "web development"},
            experience_years=5,
            location="Remote",
            job_type="Full-time",
            salary_range=(100000, 150000)
        )
    }
    
    # Calculate industry weights
    dm._calculate_industry_weights()
    
    # Get weights for technology industry
    weights = dm.get_industry_skill_weights("Technology")
    
    # Test that weights exist and are normalized
    assert len(weights) > 0
    assert all(0 <= w <= 1 for w in weights.values())
    assert "python" in weights
    assert "javascript" in weights
    
    # Test default empty dict for unknown industry
    unknown_weights = dm.get_industry_skill_weights("NonexistentIndustry")
    assert unknown_weights == {}

def test_split_text_to_list(data_manager):
    """Test text splitting functionality"""
    dm = data_manager
    
    # Test semicolon splitting
    text = "First item; Second item; Third item"
    result = dm._split_text_to_list(text)
    assert len(result) == 3
    assert "First item" in result
    assert "Second item" in result
    assert "Third item" in result
    
    # Test sentence splitting
    text = "First sentence. Second sentence. Third sentence."
    result = dm._split_text_to_list(text)
    assert len(result) == 3
    assert "First sentence" in result
    assert "Second sentence" in result
    assert "Third sentence" in result
    
    # Test empty input
    assert dm._split_text_to_list("") == []
    assert dm._split_text_to_list(None) == []
    
    # Test single sentence
    assert dm._split_text_to_list("Single sentence") == ["Single sentence"]
    
    # Test mixed delimiters
    text = "First item; Second item. Third item; Fourth item."
    result = dm._split_text_to_list(text)
    assert len(result) == 4
    assert "First item" in result
    assert "Second item" in result
    assert "Third item" in result
    assert "Fourth item" in result
    assert dm._split_text_to_list("") == []
    
def test_build_skill_patterns(data_manager):
    """Test skill pattern building with abbreviations"""
    dm = data_manager
    
    # Add test skills with common abbreviations
    dm.job_requirements = {
        "test": JobRequirements(
            title="Test",
            industry="Technology",
            responsibilities=["Develop ML and DL models using Python"],
            qualifications=["AWS and NLP experience"],
            required_skills={
                "Machine Learning",
                "Deep Learning",
                "Python",
                "Natural Language Processing",
                "Amazon Web Services",
                "JavaScript"
            },
            experience_years=3,
            location="Remote",
            job_type="Full-time",
            salary_range=(90000, 130000)
        )
    }
    
    # Build patterns
    dm._build_skill_patterns()
    
    # Test Machine Learning variations
    ml_variations = dm.skill_patterns["Machine Learning"]
    assert "machine learning" in ml_variations
    assert "machine-learning" in ml_variations
    assert "machinelearning" in ml_variations
    assert "ml" in ml_variations
    
    # Test Python variations
    python_variations = dm.skill_patterns["Python"]
    assert "python" in python_variations
    assert "py" in python_variations
    
    # Test Natural Language Processing variations
    nlp_variations = dm.skill_patterns["Natural Language Processing"]
    assert "natural language processing" in nlp_variations
    assert "nlp" in nlp_variations
    
    # Test AWS variations
    aws_variations = dm.skill_patterns["Amazon Web Services"]
    assert "amazon web services" in aws_variations
    assert "aws" in aws_variations
    
    # Test Deep Learning variations
    dl_variations = dm.skill_patterns["Deep Learning"]
    assert "deep learning" in dl_variations
    assert "deep-learning" in dl_variations
    assert "deeplearning" in dl_variations
    assert "dl" in dl_variations
    
    # Test JavaScript variations
    js_variations = dm.skill_patterns["JavaScript"]
    assert "javascript" in js_variations
    assert "js" in js_variations

def test_error_handling(data_manager):
    """Test error handling in DataManager"""
    dm = data_manager
    
    # Test invalid salary range
    assert dm._parse_salary_range("Invalid salary") == (0.0, 0.0)
    assert dm._parse_salary_range("$$$") == (0.0, 0.0)
    
    # Test invalid experience text
    assert dm._extract_experience_years("No numbers here") == 0
    assert dm._extract_experience_years("Experience: TBD") == 0
    
    # Test empty skill extraction
    assert len(dm._extract_skills("")) == 0
    
    # Test invalid text splitting
    assert dm._split_text_to_list(None) == []
    assert dm._split_text_to_list("") == []

def test_skill_pattern_persistence(tmp_path, data_manager):
    """Test saving and loading skill patterns"""
    dm = data_manager
    
    # Add test skills
    dm.job_requirements = {
        "test": JobRequirements(
            title="Test",
            industry="Technology",
            responsibilities=["Develop ML models"],
            qualifications=["Python expertise"],
            required_skills={
                "Machine Learning",
                "Deep Learning",
                "Python",
                "Natural Language Processing",
                "JavaScript",
                "Amazon Web Services"
            },
            experience_years=3,
            location="Remote",
            job_type="Full-time",
            salary_range=(90000, 130000)
        )
    }
    
    # Build and save patterns
    dm._build_skill_patterns()
    patterns_file = tmp_path / "skill_patterns.json"
    dm.save_skill_patterns(str(patterns_file))
    
    # Create new instance and load patterns directly
    new_dm = DataManager(str(tmp_path))
    new_dm.skill_patterns = {}
    new_dm.load_skill_patterns(str(patterns_file))
    
    # Verify patterns were preserved
    assert len(new_dm.skill_patterns) == len(dm.skill_patterns)
    
    # Test Machine Learning variations
    ml_variations = new_dm.skill_patterns["Machine Learning"]
    assert "machine learning" in ml_variations
    assert "machine-learning" in ml_variations
    assert "ml" in ml_variations
    
    # Test Python variations
    python_variations = new_dm.skill_patterns["Python"]
    assert "python" in python_variations
    assert "py" in python_variations
    
    # Test AWS variations
    aws_variations = new_dm.skill_patterns["Amazon Web Services"]
    assert "amazon web services" in aws_variations
    assert "aws" in aws_variations
    new_dm.load_skill_patterns(str(patterns_file))
    
    # Verify patterns were preserved
    assert len(new_dm.skill_patterns) == len(dm.skill_patterns)
    assert "Machine Learning" in new_dm.skill_patterns
    assert "Python" in new_dm.skill_patterns
    assert "Deep Learning" in new_dm.skill_patterns
    
    # Verify variations were preserved
    ml_variations = new_dm.skill_patterns["Machine Learning"]
    assert "machine learning" in ml_variations
    assert "ML" in ml_variations
    assert "machine-learning" in ml_variations
    
    assert "python" in dm.skill_patterns["Python"]
    assert "PYTHON" in dm.skill_patterns["Python"]
