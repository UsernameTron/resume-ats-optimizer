from typing import Set, Dict, List, Tuple
from .base_matcher import BaseMatcher
from .pattern_matcher import PatternMatcher
from .semantic_matcher import SemanticMatcher
from .skill_graph import SkillGraph

class CombinedMatcher(BaseMatcher):
    """Combined matcher using pattern matching, semantic similarity, and skill relationships"""
    
    def __init__(self, industry_skills: Dict[str, Dict[str, float]], 
                 cache_dir: str = "cache",
                 data_dir: str = "data"):
        super().__init__()
        
        # Initialize component matchers
        self.pattern_matcher = PatternMatcher(industry_skills)
        self.semantic_matcher = SemanticMatcher(cache_dir)
        self.skill_graph = SkillGraph(data_dir)
        
        # Weights for different matching components
        self.weights = {
            'pattern': 0.4,
            'semantic': 0.4,
            'graph': 0.2
        }
    
    def match(self, source_skills: Set[str], target_skills: Set[str]) -> float:
        """Match skills using all available methods"""
        if not source_skills or not target_skills:
            return 0.0
            
        # Get scores from each matcher
        pattern_score = self.pattern_matcher.match(source_skills, target_skills)
        semantic_score = self.semantic_matcher.match(source_skills, target_skills)
        graph_score = self._calculate_graph_score(source_skills, target_skills)
        
        # Calculate weighted average
        final_score = (
            pattern_score * self.weights['pattern'] +
            semantic_score * self.weights['semantic'] +
            graph_score * self.weights['graph']
        )
        
        self.logger.debug(
            f"Match scores - Pattern: {pattern_score:.2f}, "
            f"Semantic: {semantic_score:.2f}, "
            f"Graph: {graph_score:.2f}, "
            f"Final: {final_score:.2f}"
        )
        
        return final_score
    
    def _calculate_graph_score(self, source_skills: Set[str], 
                             target_skills: Set[str]) -> float:
        """Calculate score based on skill graph relationships"""
        if not source_skills or not target_skills:
            return 0.0
            
        total_scores = []
        
        for source_skill in source_skills:
            skill_scores = []
            
            for target_skill in target_skills:
                # Check if skills are related in the graph
                if self.skill_graph.are_skills_related(source_skill, target_skill):
                    related_skills = self.skill_graph.get_related_skills(source_skill)
                    similarity = related_skills.get(target_skill, 0.0)
                    skill_scores.append(similarity)
            
            if skill_scores:
                # Use maximum similarity for this source skill
                total_scores.append(max(skill_scores))
        
        # Return average of best matches
        return sum(total_scores) / len(source_skills) if total_scores else 0.0
    
    def extract_skills(self, text: str) -> Set[str]:
        """Extract skills using all available methods"""
        # Get skills from each method
        pattern_skills = self.pattern_matcher.extract_skills(text)
        semantic_skills = self.semantic_matcher.extract_skills(text)
        
        # Combine skills
        all_skills = pattern_skills.union(semantic_skills)
        
        # Expand skills using skill graph
        expanded_skills = set()
        for skill in all_skills:
            expanded_skills.add(skill)
            # Add synonyms
            expanded_skills.update(self.skill_graph.get_synonyms(skill))
            # Add parent skills
            expanded_skills.update(self.skill_graph.get_parents(skill))
        
        return expanded_skills
    
    def get_skill_weight(self, skill: str) -> float:
        """Get skill weight using all available methods"""
        weights = []
        
        # Get weights from each method
        pattern_weight = self.pattern_matcher.get_skill_weight(skill)
        semantic_weight = self.semantic_matcher.get_skill_weight(skill)
        
        # Add weights with component weights
        weights.append(pattern_weight * self.weights['pattern'])
        weights.append(semantic_weight * self.weights['semantic'])
        
        # Add graph-based weight
        related_skills = self.skill_graph.get_related_skills(skill)
        if related_skills:
            graph_weight = max(related_skills.values()) * 3.0  # Scale to 0.5-3.0 range
            weights.append(graph_weight * self.weights['graph'])
        
        # Return weighted average
        return sum(weights)
