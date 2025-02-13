import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture(scope="session")
def test_data():
    """Shared test data"""
    return {
        "sample_resume": """
        Senior Software Engineer with 5+ years experience
        Skills: Python, JavaScript, React.js, SQL
        2018 - present: Lead Developer at Tech Corp
        2015 - 2018: Software Engineer at StartupCo
        """,
        
        "sample_job": """
        Senior Software Engineer
        Required: 5+ years experience in Python and JavaScript
        Salary Range: $120k - $180k
        Location: Remote
        """
    }
