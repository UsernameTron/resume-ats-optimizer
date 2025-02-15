import re
from typing import Set, Dict, List, Tuple
from .base_matcher import BaseMatcher
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class PatternMatcher(BaseMatcher):
    """Pattern-based skill matcher using regex and NLTK"""
    
    def __init__(self, industry_skills: Dict[str, Dict[str, float]], stop_words: Set[str] = None):
        super().__init__()
        self.industry_skills = industry_skills
        self.stop_words = stop_words or set(stopwords.words('english'))
        self.patterns = self._build_patterns()
        
    def _build_patterns(self) -> List[Tuple[str, float]]:
        """Build regex patterns for skill matching with weights"""
        patterns = []
        
        # Core CS/CX patterns with high weights
        core_patterns = [
            (r'\b(customer|client)\s*(success|experience|support)\b', 3.0),
            (r'\b(account|relationship)\s*management\b', 2.5),
            (r'\b(churn|retention)\s*(prevention|management)\b', 2.5),
            (r'\bcx\b|\bcs\b', 2.0),
        ]
        patterns.extend(core_patterns)
        
        # Tool-specific patterns
        tool_patterns = [
            (r'\b(salesforce|gainsight|zendesk)\b', 2.0),
            (r'\b(nps|csat|ces)\b', 1.75),
            (r'\b(jira|confluence|asana)\b', 1.5),
        ]
        patterns.extend(tool_patterns)
        
        # Metric patterns
        metric_patterns = [
            (r'\b(revenue|growth|expansion)\b', 1.75),
            (r'\b(adoption|engagement|satisfaction)\b', 1.5),
            (r'\b(reporting|analytics|kpis)\b', 1.5),
        ]
        patterns.extend(metric_patterns)
        
        return patterns
    
    def match(self, source_skills: Set[str], target_skills: Set[str]) -> float:
        """Match skills using pattern-based approach"""
        if not source_skills or not target_skills:
            return 0.0
            
        total_weight = 0.0
        matched_weight = 0.0
        
        for skill in source_skills:
            weight = self.get_skill_weight(skill)
            total_weight += weight
            
            if any(self._is_skill_match(skill, target_skill) 
                  for target_skill in target_skills):
                matched_weight += weight
                
        return (matched_weight / total_weight) if total_weight > 0 else 0.0
    
    def _is_skill_match(self, skill1: str, skill2: str) -> bool:
        """Check if two skills match using pattern-based comparison"""
        skill1, skill2 = skill1.lower(), skill2.lower()
        
        # Direct match
        if skill1 == skill2:
            return True
            
        # Pattern-based match
        for pattern, _ in self.patterns:
            if (re.search(pattern, skill1) and re.search(pattern, skill2)):
                return True
                
        return False
    
    def extract_skills(self, text: str) -> Set[str]:
        """Extract skills from text using patterns"""
        skills = set()
        text = text.lower()
        
        # Tokenize and remove stop words
        words = [w for w in word_tokenize(text) if w not in self.stop_words]
        text_clean = ' '.join(words)
        
        # Extract skills using patterns
        for pattern, weight in self.patterns:
            matches = re.finditer(pattern, text_clean)
            for match in matches:
                skill = match.group().strip()
                if len(skill) > 2:  # Avoid single letters/numbers
                    skills.add(skill)
        
        return skills
    
    def get_skill_weight(self, skill: str) -> float:
        """Get weight for a skill based on patterns and industry data"""
        skill = skill.lower()
        
        # Check pattern weights
        for pattern, weight in self.patterns:
            if re.search(pattern, skill):
                return weight
                
        # Fallback to industry weights
        for industry_skills in self.industry_skills.values():
            if skill in industry_skills:
                return industry_skills[skill]
                
        return 1.0  # Default weight
