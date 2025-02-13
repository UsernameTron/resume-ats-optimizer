import torch
import numpy as np
from typing import Dict, List, Set, Optional
import spacy
from dataclasses import dataclass
import logging
from ..utils.monitoring import ResourceMonitor
from ..utils.text_processor import TextProcessor

@dataclass
class AnalysisResult:
    ats_score: float
    keyword_density: float
    matched_keywords: Dict[str, float]
    missing_keywords: List[str]
    section_scores: Dict[str, float]

class ATSAnalyzer:
    def __init__(self):
        # Initialize with MPS/Metal acceleration if available
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.nlp = spacy.load("en_core_web_sm")
        self.text_processor = TextProcessor()
        self.monitor = ResourceMonitor()
        self.logger = logging.getLogger(__name__)

    def analyze_resume(self, resume_text: str, job_description: str) -> AnalysisResult:
        """
        Analyze resume against job description with hardware-optimized processing.
        """
        try:
            with self.monitor.track_performance():
                # Process texts
                resume_doc = self.nlp(resume_text)
                job_doc = self.nlp(job_description)

                # Extract and analyze keywords
                job_keywords = self._extract_keywords(job_doc)
                resume_keywords = self._extract_keywords(resume_doc)
                
                # Calculate scores
                ats_score = self._calculate_ats_score(resume_keywords, job_keywords)
                keyword_density = self._calculate_keyword_density(resume_text, resume_keywords)
                section_scores = self._analyze_sections(resume_doc)
                
                # Get keyword matches and missing keywords
                matched_keywords = self._get_keyword_matches(resume_keywords, job_keywords)
                missing_keywords = list(job_keywords - resume_keywords)

                return AnalysisResult(
                    ats_score=ats_score,
                    keyword_density=keyword_density,
                    matched_keywords=matched_keywords,
                    missing_keywords=missing_keywords,
                    section_scores=section_scores
                )

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise

    def _extract_keywords(self, doc) -> Set[str]:
        """Extract relevant keywords using SpaCy's token attributes."""
        keywords = set()
        for token in doc:
            if (not token.is_stop and not token.is_punct and 
                token.is_alpha and len(token.text) > 2):
                keywords.add(token.text.lower())
        return keywords

    def _calculate_ats_score(self, resume_keywords: Set[str], 
                           job_keywords: Set[str]) -> float:
        """Calculate ATS match score with weighted importance."""
        if not job_keywords:
            return 0.0
        
        matches = resume_keywords & job_keywords
        return len(matches) / len(job_keywords)

    def _calculate_keyword_density(self, text: str, 
                                 keywords: Set[str]) -> float:
        """Calculate keyword density percentage."""
        words = text.lower().split()
        if not words:
            return 0.0
        
        keyword_count = sum(1 for word in words if word in keywords)
        return (keyword_count / len(words)) * 100

    def _analyze_sections(self, doc) -> Dict[str, float]:
        """Analyze resume sections for completeness and relevance."""
        sections = {
            'experience': 0.0,
            'education': 0.0,
            'skills': 0.0,
            'summary': 0.0
        }
        
        # Basic section detection and scoring
        text = doc.text.lower()
        for section in sections:
            if section in text:
                sections[section] = 1.0
                
        return sections

    def _get_keyword_matches(self, resume_keywords: Set[str], 
                           job_keywords: Set[str]) -> Dict[str, float]:
        """Get matched keywords with relevance scores."""
        matches = {}
        for keyword in (resume_keywords & job_keywords):
            # Basic relevance score for now, can be enhanced with ML
            matches[keyword] = 1.0
        return matches
