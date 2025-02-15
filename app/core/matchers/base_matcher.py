from abc import ABC, abstractmethod
from typing import Set, Dict, List, Tuple
import logging

class BaseMatcher(ABC):
    """Base class for all matchers"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def match(self, source_skills: Set[str], target_skills: Set[str]) -> float:
        """Match skills and return a similarity score"""
        pass
    
    @abstractmethod
    def extract_skills(self, text: str) -> Set[str]:
        """Extract skills from text"""
        pass
    
    @abstractmethod
    def get_skill_weight(self, skill: str) -> float:
        """Get the weight for a specific skill"""
        pass
