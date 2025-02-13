# Enhanced ATS Resume Optimizer

A high-performance resume optimization system designed for modern ATS (Applicant Tracking Systems).

## Features

- Hardware-accelerated text processing using Metal/MPS
- Real-time ATS score calculation
- Keyword density optimization
- Section-based analysis
- Resource usage monitoring
- Performance optimization

## System Requirements

- MacBook Pro M4 Pro (or compatible)
- 48GB RAM
- Python 3.9+
- Metal/MPS support for hardware acceleration

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Project Structure

```
enhanced-ats-optimizer/
├── app/
│   ├── core/          # Core analysis and optimization logic
│   ├── utils/         # Utility functions and monitoring
│   └── api/           # FastAPI routes and models
├── tests/             # Test suites
│   ├── unit/
│   └── integration/
└── config/           # Configuration files
```

## Usage

```python
from app.core.analyzer import ATSAnalyzer

analyzer = ATSAnalyzer()
result = analyzer.analyze_resume(resume_text, job_description)
print(f"ATS Score: {result.ats_score * 100}%")
print(f"Keyword Density: {result.keyword_density}%")
```

## Performance Metrics

- Processing Time: < 2.5 seconds
- Memory Usage: < 70% (33.6GB)
- Error Rate: < 5%

## Testing

Run the test suite:
```bash
python -m unittest discover tests
```
