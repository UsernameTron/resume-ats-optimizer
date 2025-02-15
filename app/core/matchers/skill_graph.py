from typing import Set, Dict, List, Tuple, Optional
import networkx as nx
from collections import defaultdict
import json
from pathlib import Path
import logging
import numpy as np
import re
from joblib import Memory

class SkillGraph:
    """Graph-based skill relationship manager"""
    
    def __init__(self, data_dir: str = "data"):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.graph = nx.Graph()
        self.synonyms = defaultdict(set)
        self.hierarchies = defaultdict(set)
        self.co_occurrences = defaultdict(lambda: defaultdict(int))
        self.skill_frequencies = defaultdict(int)
        self.total_documents = 0
        
        # Initialize cache
        cache_dir = Path(data_dir) / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory = Memory(cache_dir, verbose=0)
        
        # Initialize compiled regex patterns
        self.compiled_patterns = {}
        
        self.initialize_graph()
    
    def initialize_graph(self):
        """Initialize the skill relationship graph"""
        # Load predefined CS/CX skill relationships
        self._add_core_relationships()
        self._load_saved_relationships()
    
    def _add_core_relationships(self):
        """Add core CS/CX skill relationships"""
        # Tool relationships
        self._add_tool_relationships()
        
        # Metric relationships
        self._add_metric_relationships()
        
        # Process relationships
        self._add_process_relationships()
        
        # Soft skill relationships
        self._add_soft_skill_relationships()
    
    def compile_patterns(self, patterns: List[Tuple[str, float]]) -> Dict[str, Tuple[re.Pattern, float]]:
        """Precompile regex patterns for efficient matching"""
        compiled = {}
        for pattern, weight in patterns:
            try:
                compiled[pattern] = (re.compile(pattern, re.IGNORECASE), weight)
            except re.error as e:
                self.logger.error(f"Failed to compile pattern {pattern}: {str(e)}")
        return compiled

    def _add_tool_relationships(self):
        """Add tool-related skill relationships"""
        # CRM tools with regex patterns
        crm_tools = {
            'salesforce': [
                (r'\b(salesforce|sfdc|sales\s*force|force\.com)\b', 1.0)
            ],
            'gainsight': [
                (r'\b(gainsight|gain\s*sight|gs)\b', 1.0)
            ],
            'zendesk': [
                (r'\b(zendesk|zen\s*desk)\b', 1.0)
            ],
            'hubspot': [
                (r'\b(hubspot|hub\s*spot)\b', 1.0)
            ],
        }
        
        # Compile and store patterns
        for tool, patterns in crm_tools.items():
            self.compiled_patterns[tool] = self.compile_patterns(patterns)
            # Add relationships to graph
            for pattern, weight in patterns:
                self.add_synonyms(tool, [pattern])
        
        for tool, synonyms in crm_tools.items():
            self.add_synonyms(tool, synonyms)
            self.add_parent('crm tools', tool)
    
    def _add_metric_relationships(self):
        """Add metric-related skill relationships"""
        # Customer metrics with regex patterns
        metrics = {
            'nps': [
                (r'\b(nps|net\s*promoter\s*score)\b', 1.0),
                (r'\bcustomer\s*satisfaction\b', 0.8)
            ],
            'csat': [
                (r'\b(csat|customer\s*satisfaction\s*score)\b', 1.0),
                (r'\bsatisfaction\s*metrics\b', 0.8)
            ],
            'churn': [
                (r'\b(churn\s*rate|customer\s*churn)\b', 1.0),
                (r'\bretention\s*rate\b', 0.9)
            ],
            'mrr': [
                (r'\b(mrr|monthly\s*recurring\s*revenue)\b', 1.0),
                (r'\brecurring\s*revenue\b', 0.9)
            ],
        }
        
        # Compile and store patterns
        for metric, patterns in metrics.items():
            self.compiled_patterns[metric] = self.compile_patterns(patterns)
            # Add relationships to graph
            for pattern, weight in patterns:
                self.add_synonyms(metric, [pattern])
        
        for metric, synonyms in metrics.items():
            self.add_synonyms(metric, synonyms)
            self.add_parent('customer metrics', metric)
    
    def _add_process_relationships(self):
        """Add process-related skill relationships"""
        processes = {
            'onboarding': ['customer onboarding', 'client onboarding'],
            'implementation': ['product implementation', 'solution implementation'],
            'qbr': ['quarterly business review', 'business review'],
        }
        
        for process, synonyms in processes.items():
            self.add_synonyms(process, synonyms)
            self.add_parent('customer success processes', process)
    
    def _add_soft_skill_relationships(self):
        """Add soft skill relationships"""
        soft_skills = {
            'communication': ['verbal communication', 'written communication'],
            'leadership': ['team leadership', 'project leadership'],
            'problem solving': ['troubleshooting', 'issue resolution'],
        }
        
        for skill, synonyms in soft_skills.items():
            self.add_synonyms(skill, synonyms)
            self.add_parent('soft skills', skill)
    
    def _load_saved_relationships(self):
        """Load saved skill relationships from file"""
        relationships_file = self.data_dir / "skill_relationships.json"
        if relationships_file.exists():
            try:
                with open(relationships_file, 'r') as f:
                    data = json.load(f)
                    
                # Load synonyms
                for skill, syns in data.get('synonyms', {}).items():
                    self.add_synonyms(skill, syns)
                    
                # Load hierarchies
                for parent, children in data.get('hierarchies', {}).items():
                    for child in children:
                        self.add_parent(parent, child)
                        
                self.logger.info("Loaded saved skill relationships")
            except Exception as e:
                self.logger.error(f"Failed to load skill relationships: {str(e)}")
    
    def save_relationships(self):
        """Save skill relationships to file"""
        relationships_file = self.data_dir / "skill_relationships.json"
        try:
            data = {
                'synonyms': {k: list(v) for k, v in self.synonyms.items()},
                'hierarchies': {k: list(v) for k, v in self.hierarchies.items()}
            }
            
            with open(relationships_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info("Saved skill relationships")
        except Exception as e:
            self.logger.error(f"Failed to save skill relationships: {str(e)}")
    
    def add_synonyms(self, skill: str, synonyms: List[str]):
        """Add synonym relationships for a skill"""
        skill = skill.lower()
        self.synonyms[skill].update(s.lower() for s in synonyms)
        
        # Add edges to graph
        for syn in synonyms:
            self.graph.add_edge(skill, syn.lower(), relationship='synonym')
    
    def add_parent(self, parent: str, child: str):
        """Add parent-child relationship between skills"""
        parent, child = parent.lower(), child.lower()
        self.hierarchies[parent].add(child)
        
        # Add edge to graph
        self.graph.add_edge(parent, child, relationship='parent')
    
    def get_synonyms(self, skill: str) -> Set[str]:
        """Get all synonyms for a skill"""
        skill = skill.lower()
        return self.synonyms.get(skill, set())
    
    def get_parents(self, skill: str) -> Set[str]:
        """Get all parent skills"""
        skill = skill.lower()
        return {parent for parent, children in self.hierarchies.items() 
                if skill in children}
    
    def get_children(self, skill: str) -> Set[str]:
        """Get all child skills"""
        skill = skill.lower()
        return self.hierarchies.get(skill, set())
    
    def get_related_skills(self, skill: str, max_distance: int = 2) -> Dict[str, float]:
        """Get related skills with similarity scores based on graph distance and co-occurrence"""
        skill = skill.lower()
        if skill not in self.graph:
            return {}
            
        # Use cached implementation
        get_related_impl = self._memory.cache(self._get_related_skills_impl)
        return get_related_impl(skill, max_distance)
        
    def _get_related_skills_impl(self, skill: str, max_distance: int = 2) -> Dict[str, float]:
            
        related_skills = {}
        
        # Get all paths from skill to other nodes
        for target in self.graph.nodes():
            if target == skill:
                continue
                
            try:
                # Calculate graph distance similarity
                distance = nx.shortest_path_length(self.graph, skill, target)
                if distance <= max_distance:
                    # Convert distance to similarity score (closer = higher similarity)
                    graph_similarity = 1.0 / (1.0 + distance)
                    
                    # Calculate co-occurrence similarity
                    co_occurrence_count = self.co_occurrences[skill][target]
                    total_occurrences = max(self.skill_frequencies[skill], 1)
                    co_occurrence_similarity = co_occurrence_count / total_occurrences
                    
                    # Combine similarities (weighted average)
                    combined_similarity = (0.7 * graph_similarity + 0.3 * co_occurrence_similarity)
                    related_skills[target] = combined_similarity
                    
                    # Use precompiled patterns for additional matching
                    if skill in self.compiled_patterns:
                        for pattern, (compiled_pattern, weight) in self.compiled_patterns[skill].items():
                            if compiled_pattern.search(target):
                                # Boost similarity score based on pattern match
                                related_skills[target] = min(1.0, related_skills[target] + weight * 0.2)
            except nx.NetworkXNoPath:
                continue
                
        return related_skills
    
    def update_co_occurrences(self, skills: Set[str]):
        """Update co-occurrence counts for a set of skills that appear together"""
        skills = {s.lower() for s in skills}
        self.total_documents += 1
        
        # Update individual skill frequencies
        for skill in skills:
            self.skill_frequencies[skill] += 1
            
            # Use precompiled patterns to find additional matches
            for category, patterns in self.compiled_patterns.items():
                for pattern, (compiled_pattern, weight) in patterns.items():
                    if compiled_pattern.search(skill):
                        self.skill_frequencies[category] += weight
                        skills.add(category)
        
        # Update co-occurrence counts
        for skill1 in skills:
            for skill2 in skills:
                if skill1 != skill2:
                    self.co_occurrences[skill1][skill2] += 1
        
        # Clear the cache for related skills
        self.memory.clear()
    
    def get_co_occurrence_weight(self, skill1: str, skill2: str) -> float:
        """Calculate weight based on co-occurrence frequency and pattern matching"""
        # Use cached implementation
        get_weight_impl = self._memory.cache(self._get_co_occurrence_weight_impl)
        return get_weight_impl(skill1, skill2)
    
    def _get_co_occurrence_weight_impl(self, skill1: str, skill2: str) -> float:
        """Implementation of get_co_occurrence_weight with caching"""
        skill1, skill2 = skill1.lower(), skill2.lower()
        
        if self.total_documents == 0:
            return 0.0
            
        # Get co-occurrence count
        co_occurrence_count = self.co_occurrences[skill1][skill2]
        
        # Calculate individual probabilities
        p_skill1 = self.skill_frequencies[skill1] / self.total_documents
        p_skill2 = self.skill_frequencies[skill2] / self.total_documents
        
        # Calculate joint probability
        p_joint = co_occurrence_count / self.total_documents
        
        # Calculate pointwise mutual information (PMI)
        weight = 0.0
        if p_joint > 0 and p_skill1 > 0 and p_skill2 > 0:
            pmi = np.log2(p_joint / (p_skill1 * p_skill2))
            # Normalize PMI to [0, 1]
            weight = max(0.0, pmi / (-np.log2(p_joint)))
        
        # Check pattern matches
        if skill1 in self.compiled_patterns:
            for pattern, (compiled_pattern, pattern_weight) in self.compiled_patterns[skill1].items():
                if compiled_pattern.search(skill2):
                    weight = max(weight, pattern_weight)
        
        if skill2 in self.compiled_patterns:
            for pattern, (compiled_pattern, pattern_weight) in self.compiled_patterns[skill2].items():
                if compiled_pattern.search(skill1):
                    weight = max(weight, pattern_weight)
        
        return weight
    
    def are_skills_related(self, skill1: str, skill2: str, threshold: float = 0.5) -> bool:
        """Check if two skills are related using both graph structure and co-occurrence"""
        skill1, skill2 = skill1.lower(), skill2.lower()
        
        # Check direct relationships
        if skill2 in self.get_synonyms(skill1):
            return True
            
        if skill2 in self.get_parents(skill1) or skill2 in self.get_children(skill1):
            return True
        
        # Get graph-based similarity
        related_skills = self.get_related_skills(skill1)
        graph_similarity = related_skills.get(skill2, 0.0)
        
        # Get co-occurrence weight
        co_occurrence_weight = self.get_co_occurrence_weight(skill1, skill2)
        
        # Combine weights (weighted average)
        combined_weight = (0.7 * graph_similarity + 0.3 * co_occurrence_weight)
        
        return combined_weight >= threshold
