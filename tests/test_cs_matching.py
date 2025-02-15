"""
Test suite for CS/CX skill matching functionality
"""
import pytest
from pathlib import Path
import json
from app.core.matchers.semantic_matcher import SemanticMatcher
from app.validation.cs_validator import CSValidator, ValidationMetrics

# Test data
TEST_JOB_DATA = [
    {
        'description': """Senior Customer Success Manager
        - Drive customer success and retention
        - Manage enterprise accounts using Salesforce and Gainsight
        - Track NPS, CSAT, and churn metrics
        - Lead quarterly business reviews""",
        'skills': {
            'customer success',
            'account management',
            'salesforce',
            'gainsight',
            'nps',
            'csat',
            'churn analysis'
        }
    },
    {
        'description': """Customer Experience Lead
        - Improve customer satisfaction and retention
        - Use Zendesk for support management
        - Monitor customer health scores
        - Analyze MRR and churn rates""",
        'skills': {
            'customer experience',
            'customer satisfaction',
            'zendesk',
            'customer health',
            'mrr',
            'churn rate'
        }
    }
]

TEST_RESUME_DATA = [
    {
        'content': """Experienced Customer Success Professional
        - Led enterprise customer success team
        - Expert in Salesforce, Gainsight, and Zendesk
        - Reduced churn by 25% through proactive engagement
        - Maintained 95% CSAT score""",
        'skills': {
            'customer success',
            'enterprise accounts',
            'salesforce',
            'gainsight',
            'zendesk',
            'churn reduction',
            'csat'
        }
    }
]

@pytest.fixture
def semantic_matcher():
    """Initialize semantic matcher with test cache"""
    return SemanticMatcher(cache_dir="tests/cache")

@pytest.fixture
def cs_validator():
    """Initialize CS validator with test data"""
    validator = CSValidator(data_dir="tests/data")
    # Add test data
    validator.validation_data = {
        'jobs': [(job['description'], job['skills']) for job in TEST_JOB_DATA],
        'resumes': [(resume['content'], resume['skills']) for resume in TEST_RESUME_DATA]
    }
    return validator

def test_cs_specific_matching(semantic_matcher):
    """Test CS/CX-specific skill matching"""
    # Test exact matches
    assert semantic_matcher.match(
        {"customer success", "account management"},
        {"customer success", "client management"}
    ) > 0.8
    
    # Test variations
    assert semantic_matcher.match(
        {"customer success manager", "client success"},
        {"csm", "customer success"}
    ) > 0.7
    
    # Test domain-specific terms
    assert semantic_matcher.match(
        {"churn reduction", "customer retention"},
        {"reduce churn", "retain customers"}
    ) > 0.7

def test_tool_matching(semantic_matcher):
    """Test matching of CS/CX tools"""
    # Test exact tool matches
    assert semantic_matcher.match(
        {"salesforce", "gainsight"},
        {"sfdc", "gainsight"}
    ) > 0.8
    
    # Test tool variations
    assert semantic_matcher.match(
        {"salesforce crm", "zendesk support"},
        {"sales force", "zen desk"}
    ) > 0.7

def test_metric_matching(semantic_matcher):
    """Test matching of CS/CX metrics"""
    # Test exact metric matches
    assert semantic_matcher.match(
        {"nps", "csat", "churn rate"},
        {"net promoter score", "customer satisfaction", "churn"}
    ) > 0.8
    
    # Test metric variations
    assert semantic_matcher.match(
        {"monthly recurring revenue", "customer health"},
        {"mrr", "health score"}
    ) > 0.7

def test_domain_boost(semantic_matcher):
    """Test CS/CX domain-specific boosting"""
    # Test with domain terms
    cs_similarity = semantic_matcher.match(
        {"customer success", "account management"},
        {"client success", "relationship management"}
    )
    
    # Test without domain terms
    general_similarity = semantic_matcher.match(
        {"project management", "team leadership"},
        {"manage projects", "lead teams"}
    )
    
    # Domain-specific matches should score higher
    assert cs_similarity > general_similarity

def test_validation_metrics(semantic_matcher, cs_validator):
    """Test overall validation metrics"""
    # Run validation
    metrics = cs_validator.validate_skill_extraction(semantic_matcher)
    
    # Check metrics against thresholds
    assert metrics.accuracy >= 0.85, "Accuracy below threshold"
    assert metrics.precision >= 0.87, "Precision below threshold"
    assert metrics.recall >= 0.83, "Recall below threshold"
    assert metrics.false_positive_rate <= 0.08, "False positive rate above threshold"
    
    # Check CS/CX skill coverage
    assert metrics.skill_coverage['core_cs'] >= 0.90, "Core CS skill coverage below threshold"
    assert metrics.skill_coverage['tools'] >= 0.85, "Tool coverage below threshold"
    assert metrics.skill_coverage['metrics'] >= 0.85, "Metric coverage below threshold"

def test_memory_optimization(semantic_matcher):
    """Test memory usage optimization"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Perform multiple matches
    for _ in range(100):
        semantic_matcher.match(
            {"customer success", "account management", "salesforce"},
            {"client success", "relationship management", "sfdc"}
        )
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < 50, f"Memory usage increased by {memory_increase}MB, should be < 50MB"

def test_processing_speed(semantic_matcher):
    """Test processing speed optimization"""
    import time
    
    # Test batch processing speed
    start_time = time.time()
    for _ in range(1000):
        semantic_matcher.match(
            {"customer success", "account management"},
            {"client success", "relationship management"}
        )
    total_time = time.time() - start_time
    
    docs_per_second = 1000 / total_time
    assert docs_per_second >= 500, f"Processing speed: {docs_per_second:.1f} docs/sec, should be >= 500"

def test_cache_effectiveness(semantic_matcher):
    """Test cache effectiveness"""
    # First match to populate cache
    test_skills1 = {"customer success", "account management"}
    test_skills2 = {"client success", "relationship management"}
    
    # Time first match
    import time
    start_time = time.time()
    semantic_matcher.match(test_skills1, test_skills2)
    first_match_time = time.time() - start_time
    
    # Time cached match
    start_time = time.time()
    semantic_matcher.match(test_skills1, test_skills2)
    cached_match_time = time.time() - start_time
    
    assert cached_match_time < first_match_time * 0.5, "Cache not providing significant speedup"
