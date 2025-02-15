import torch
import re
from typing import List, Set, Dict
import numpy as np

class TextProcessor:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Common resume section headers
        self.section_headers = {
            'experience': ['experience', 'work history', 'professional experience'],
            'education': ['education', 'academic background', 'qualifications'],
            'skills': ['skills', 'technical skills', 'competencies'],
            'summary': ['summary', 'professional summary', 'profile']
        }
        
    def preprocess_text(self, text: str) -> Dict[str, any]:
        """Clean and normalize text with structured output.
        
        Returns:
            Dict containing:
            - clean_text: cleaned and normalized text
            - entities: extracted named entities
            - noun_phrases: extracted noun phrases
            - technical_terms: identified technical terms
            - sections: extracted document sections
        """
        try:
            # Basic text cleaning
            clean_text = text.lower()
            clean_text = re.sub(r'\s+', ' ', clean_text)
            clean_text = re.sub(r'[^\w\s]', '', clean_text)
            clean_text = clean_text.strip()
            
            # Extract sections
            sections = self.extract_sections(text)
            
            # Extract technical terms (skills, tools, etc.)
            technical_terms = set()
            for section in sections.values():
                # Look for terms that might be technical (e.g., capitalized terms, terms with numbers)
                tech_matches = re.findall(r'\b[A-Z][A-Za-z0-9.#+]+\b', section)
                technical_terms.update(tech_matches)
            
            # Extract potential entities (proper nouns)
            entities = set()
            entity_matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            entities.update(entity_matches)
            
            # Extract noun phrases (using simple patterns for now)
            noun_phrases = set()
            np_matches = re.findall(r'\b(?:the\s+)?(?:new\s+)?(?:big\s+)?[A-Za-z]+(?:\s+[A-Za-z]+){0,2}\b', clean_text)
            noun_phrases.update(np_matches)
            
            return {
                'clean_text': clean_text,
                'entities': list(entities),
                'noun_phrases': list(noun_phrases),
                'technical_terms': list(technical_terms),
                'sections': sections
            }
            
        except Exception as e:
            # Log the error and return a basic structure
            print(f"Error in text preprocessing: {str(e)}")
            return {
                'clean_text': text.lower().strip(),
                'entities': [],
                'noun_phrases': [],
                'technical_terms': [],
                'sections': {}
            }
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract different sections from the resume."""
        sections = {}
        lines = text.split('\n')
        current_section = 'other'
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            for section, headers in self.section_headers.items():
                if any(header in line_lower for header in headers):
                    current_section = section
                    sections[current_section] = []
                    break
            
            if current_section in sections:
                sections[current_section].append(line)
        
        # Convert lists to strings
        return {k: '\n'.join(v) for k, v in sections.items()}
    
    def calculate_section_weights(self, sections: Dict[str, str]) -> Dict[str, float]:
        """Calculate importance weights for each section."""
        total_length = sum(len(content) for content in sections.values())
        if total_length == 0:
            return {section: 0.0 for section in sections}
        
        return {section: len(content) / total_length 
                for section, content in sections.items()}
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return [word.strip() for word in text.split() 
                if word.strip() and len(word.strip()) > 2]
    
    def get_ngrams(self, text: str, n: int = 2) -> List[str]:
        """Generate n-grams from text."""
        tokens = self.tokenize(text)
        return [' '.join(tokens[i:i+n]) 
                for i in range(len(tokens)-n+1)]
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using token overlap."""
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union)
