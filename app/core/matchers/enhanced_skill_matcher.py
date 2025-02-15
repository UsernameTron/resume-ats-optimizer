"""
Enhanced skill matcher with improved precision and recall for CS/CX domain
"""
from typing import Set, Dict, List, Tuple, Optional
from pathlib import Path
import re
import logging
from collections import defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from .base_matcher import BaseMatcher
from .cs_metric_detector import CSMetricDetector, MetricMatch

class EnhancedSkillMatcher(BaseMatcher):
    def __init__(self, cache_dir: str = "cache"):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize base components
        self.lemmatizer = WordNetLemmatizer()
        self.logger = logging.getLogger(__name__)
        
        # Enhanced confidence thresholds with domain focus
        self.base_confidence = 0.65  # Lowered base confidence for better recall
        
        # Context multipliers with refined weights
        self.context_multipliers = {
            'direct_mention': 1.8,    # Increased for explicit mentions
            'achievement': 1.6,       # Boosted for proven skills
            'tool_usage': 1.7,        # Higher for practical experience
            'responsibility': 1.5,    # Increased for role-specific skills
            'quantified': 1.7,        # Increased for metrics-backed skills
            'leadership': 1.6,        # Increased for management skills
            'customer_interaction': 1.8,  # New: for customer-facing skills
            'technical_expertise': 1.7,   # New: for technical proficiency
            'process_improvement': 1.6,   # New: for optimization skills
            'team_collaboration': 1.5     # New: for collaborative skills
        }
        
        # Refined skill-specific adjustments
        self.skill_confidence_adjustments = {
            'customer_service': {
                'base_boost': 1.5,     # Increased for core CS skills
                'context_requirement': 0.65  # Lowered threshold
            },
            'technical_support': {
                'base_boost': 1.6,     # Higher for technical skills
                'context_requirement': 0.70
            },
            'account_management': {
                'base_boost': 1.5,     # Increased for key function
                'context_requirement': 0.65
            },
            'data_analysis': {
                'base_boost': 1.4,     # Increased for support role
                'context_requirement': 0.75
            },
            'leadership': {
                'base_boost': 1.4,     # Increased for leadership
                'context_requirement': 0.70
            },
            'communication': {
                'base_boost': 1.4,     # Increased for soft skills
                'context_requirement': 0.65
            },
            'crm_tools': {
                'base_boost': 1.6,     # New: for CRM expertise
                'context_requirement': 0.70
            },
            'process_optimization': {
                'base_boost': 1.5,     # New: for process skills
                'context_requirement': 0.70
            },
            'stakeholder_management': {
                'base_boost': 1.5,     # New: for relationship skills
                'context_requirement': 0.70
            },
            'project_management': {
                'base_boost': 1.4,     # New: for project skills
                'context_requirement': 0.75
            }
        }
        
        # Initialize enhanced stopwords
        self.stop_words = self._get_domain_stopwords()
        
        # Initialize vectorizer with optimized parameters
        self.vectorizer = TfidfVectorizer(
            analyzer=self._custom_analyzer,
            ngram_range=(1, 3),
            min_df=1,
            max_df=1.0,
            max_features=15000,
            sublinear_tf=True
        )
        
        # Initialize caches
        self.vectors_cache = {}
        self.context_cache = {}
        
        # Initialize CS metric detector
        self.metric_detector = CSMetricDetector()
        
    def _get_domain_stopwords(self) -> Set[str]:
        """Get domain-specific stopwords"""
        base_stopwords = set(stopwords.words('english'))
        domain_stopwords = {
            'experience', 'skill', 'skills', 'year', 'years',
            'knowledge', 'understanding', 'ability', 'proficiency',
            'familiar', 'familiarity', 'basic', 'advanced', 'intermediate',
            'level', 'levels', 'certification', 'certified', 'degree'
        }
        return base_stopwords.union(domain_stopwords)
        
    def extract_skills(self, text: str) -> Set[str]:
        """Enhanced skill extraction with improved precision and recall"""
        if not text:
            return set()
            
        # Cache key for this text
        cache_key = f"skills_{hash(text)}"
        if cache_key in self.vectors_cache:
            return self.vectors_cache[cache_key]
            
        # Extract candidates using multiple approaches
        candidates = set()
        
        # TF-IDF candidates
        tfidf_candidates = self._extract_candidates(text)
        candidates.update(tfidf_candidates)
        
        # Pattern-based candidates
        pattern_candidates = self._extract_pattern_candidates(text)
        candidates.update(pattern_candidates)
        
        # Validate candidates
        validated_skills = self._validate_skills(candidates, text)
        
        # Analyze context for each skill
        skill_info = []
        sentences = sent_tokenize(text)
        for skill in validated_skills:
            if isinstance(skill, dict):
                skill_text = skill['skill']
            else:
                skill_text = skill
                
            contexts = self._find_skill_contexts(skill_text, sentences)
            context_score = self._analyze_skill_context(skill_text, contexts)
            confidence = self._calculate_confidence(skill_text, context_score, contexts)
            skill_info.append({
                'skill': skill_text,
                'contexts': contexts,
                'confidence': confidence
            })
        
        # Apply domain boosts
        boosted_skills = self._apply_domain_boosts(skill_info, text)
        
        # Filter by confidence threshold
        final_skills = {skill['skill'] for skill in boosted_skills 
                       if skill['confidence'] >= self.base_confidence}
        
        # Cache results
        self.vectors_cache[cache_key] = final_skills
        return final_skills
        
    def _extract_pattern_candidates(self, text: str) -> Set[str]:
        """Extract skill candidates using pattern matching"""
        candidates = set()
        sentences = sent_tokenize(text)
        
        # Common skill patterns
        patterns = [
            # Tool/technology patterns
            r'(?:using|utilized|implemented|configured|administered)\s+(\w+(?:\s+\w+){0,2})',
            
            # Achievement patterns
            r'(?:improved|optimized|enhanced|streamlined|transformed)\s+(\w+(?:\s+\w+){0,2})',
            
            # Responsibility patterns
            r'(?:responsible for|managed|led|coordinated|oversaw)\s+(\w+(?:\s+\w+){0,2})',
            
            # Expertise patterns
            r'(?:expertise in|proficient in|skilled in|specialized in)\s+(\w+(?:\s+\w+){0,2})',
            
            # Customer service patterns
            r'(?:customer|client)\s+(?:success|support|service|experience|satisfaction|engagement)\s+(\w+(?:\s+\w+){0,2})',
            r'(?:managed|handled|resolved)\s+(?:customer|client)\s+(\w+(?:\s+\w+){0,2})',
            
            # Process patterns
            r'(?:process|workflow|system|protocol)\s+(\w+(?:\s+\w+){0,2})',
            
            # Communication patterns
            r'(?:communicated|presented|documented)\s+(\w+(?:\s+\w+){0,2})',
            
            # Analysis patterns
            r'(?:analyzed|monitored|tracked|measured)\s+(\w+(?:\s+\w+){0,2})',
            
            # Training patterns
            r'(?:trained|mentored|coached|guided)\s+(\w+(?:\s+\w+){0,2})',
            
            # Strategy patterns
            r'(?:developed|implemented|executed)\s+(?:strategy|plan|program|initiative)\s+for\s+(\w+(?:\s+\w+){0,2})',
            
            # Metric patterns
            r'(?:improved|increased|reduced)\s+(?:\d+%|\$\d+[kKmMbB]?|\d+x)\s+(\w+(?:\s+\w+){0,2})',
            
            # Tool expertise patterns
            r'(?:proficient|experienced|skilled)\s+(?:with|in)\s+(\w+(?:\s+\w+){0,2})',
            
            # Project patterns
            r'(?:led|managed|executed)\s+(?:project|initiative)\s+for\s+(\w+(?:\s+\w+){0,2})'
        ]
        
        for sentence in sentences:
            for pattern in patterns:
                matches = re.finditer(pattern, sentence.lower())
                for match in matches:
                    skill = match.group(1).strip()
                    if len(skill) > 2 and skill not in self.stop_words:
                        candidates.add(skill)
        
        return candidates
                
    def _extract_candidates(self, text: str) -> List[str]:
        """Extract initial skill candidates using TF-IDF and domain patterns"""
        candidates = set()
        
        # TF-IDF based extraction
        processed_text = self._preprocess_text(text)
        tfidf_matrix = self.vectorizer.fit_transform([processed_text])
        feature_names = self.vectorizer.get_feature_names_out()
        non_zero = tfidf_matrix.nonzero()
        scores = zip(non_zero[1], tfidf_matrix.data)
        
        # Add TF-IDF candidates
        for idx, score in scores:
            if score >= 0.1:
                candidates.add(feature_names[idx])
        
        # CS/CX specific skill patterns
        cs_patterns = [
            # Customer interaction skills
            r'(?:customer|client)\s+(?:service|support|success|experience)\s+(?:specialist|representative|agent)',
            r'(?:account|relationship)\s+(?:management|manager|executive)',
            r'(?:technical|product)\s+(?:support|specialist|consultant)',
            
            # Communication skills
            r'(?:verbal|written)\s+communication',
            r'(?:presentation|documentation)\s+skills',
            r'stakeholder\s+(?:management|communication)',
            
            # Technical skills
            r'(?:CRM|helpdesk|ticketing)\s+(?:system|software|tool)',
            r'(?:Salesforce|Zendesk|Intercom|Freshdesk|ServiceNow)',
            r'(?:troubleshooting|problem-solving)\s+(?:skills|ability)',
            
            # Process skills
            r'(?:SLA|KPI|QA)\s+(?:management|monitoring|tracking)',
            r'(?:escalation|resolution)\s+(?:process|protocol|management)',
            r'(?:workflow|process)\s+(?:optimization|improvement)',
            
            # Analytical skills
            r'(?:data|metrics|analytics)\s+(?:analysis|reporting|tracking)',
            r'(?:performance|satisfaction|NPS)\s+(?:metrics|monitoring)',
            r'(?:business|customer)\s+(?:intelligence|insights)',
            
            # Project skills
            r'(?:project|program|initiative)\s+(?:management|coordination)',
            r'(?:change|release|deployment)\s+(?:management|coordination)',
            r'(?:training|onboarding)\s+(?:program|material|development)',
            
            # Tool expertise
            r'(?:proficient|experienced)\s+(?:with|in)\s+(?:Salesforce|Zendesk|Intercom|JIRA|Confluence|ServiceNow)',
            r'(?:experience|expertise)\s+(?:with|in)\s+(?:CRM|ticketing|helpdesk)\s+(?:systems|software)',
            
            # Soft skills
            r'(?:empathy|patience|adaptability|flexibility)',
            r'(?:team|cross-functional)\s+collaboration',
            r'(?:time|priority)\s+management'
        ]
        
        for pattern in cs_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                skill = match.group(0).strip()
                if len(skill) > 2 and skill not in self.stop_words:
                    candidates.add(skill)
        
        return list(candidates)
        
    def _validate_skills(self, candidates: List[str], text: str) -> List[Dict]:
        """Validate skills with enhanced context awareness"""
        validated = []
        sentences = sent_tokenize(text)
        
        for skill in candidates:
            # Skip common false positives
            if self._check_false_positive(skill, text):
                continue
                
            # Find skill mentions
            skill_contexts = self._find_skill_contexts(skill, sentences)
            if not skill_contexts:
                continue
                
            # Calculate enhanced context score
            context_score = self._analyze_skill_context(skill, skill_contexts)
            
            # Validate skill coherence
            if not self._validate_skill_coherence(skill):
                continue
                
            # Calculate confidence with context
            confidence = self._calculate_confidence(skill, context_score, skill_contexts)
            
            if confidence >= self.base_confidence:
                validated.append({
                    'skill': skill,
                    'context_score': context_score,
                    'confidence': confidence,
                    'contexts': skill_contexts
                })
                
        return validated
        
    def _find_skill_contexts(self, skill: str, sentences: List[str]) -> List[Dict]:
        """Find and analyze all contexts where a skill is mentioned"""
        contexts = []
        skill_lower = skill.lower()
        
        for sentence in sentences:
            if skill_lower in sentence.lower():
                # Analyze context characteristics
                context_info = {
                    'sentence': sentence,
                    'achievement_indicators': self._find_achievement_indicators(sentence),
                    'quantified_metrics': self._find_quantified_metrics(sentence),
                    'tool_usage': self._find_tool_usage(sentence),
                    'leadership_indicators': self._find_leadership_indicators(sentence)
                }
                contexts.append(context_info)
                
        return contexts
        
    def _analyze_metrics(self, text: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """Analyze metrics in text with enhanced CS/CX focus"""
        metrics = self.metric_detector.extract_metrics(text)
        
        # Filter high-confidence metrics
        high_confidence_metrics = [m for m in metrics if m.confidence >= 0.8]
        
        # Group metrics by category for context analysis
        metrics_by_category = {}
        for metric in high_confidence_metrics:
            if metric.category not in metrics_by_category:
                metrics_by_category[metric.category] = []
            metrics_by_category[metric.category].append(metric)
        
        # Analyze metric patterns for skill confidence boost
        has_strong_metrics = False
        if high_confidence_metrics:
            # Check for diverse metric types (multiple categories)
            if len(metrics_by_category) >= 2:
                has_strong_metrics = True
            # Or check for multiple high-confidence metrics in same category
            elif any(len(metrics) >= 2 for metrics in metrics_by_category.values()):
                has_strong_metrics = True
            # Or check for very high confidence single metric
            elif any(m.confidence >= 0.9 for m in high_confidence_metrics):
                has_strong_metrics = True
        
        # Convert metrics to dict format for context analysis
        metric_contexts = [{
            'type': 'metric',
            'category': m.category,
            'subcategory': m.subcategory,
            'value': m.value,
            'confidence': m.confidence,
            'context': m.context,
            'timeframe': m.timeframe
        } for m in high_confidence_metrics]
        
        return has_strong_metrics, metric_contexts

    def _analyze_skill_context(self, skill: str, contexts: List[Dict]) -> float:
        """Enhanced context analysis with multiple factors"""
        if not contexts:
            return 0.0
            
        context_scores = []
        for context in contexts:
            score = 1.0  # Base score
            
            # Apply multipliers based on context characteristics
            if context['achievement_indicators']:
                score *= self.context_multipliers['achievement']
            # Check for metrics with enhanced detection
            has_metrics, metric_contexts = self._analyze_metrics(context.get('text', ''))
            if has_metrics:
                score *= self.context_multipliers['quantified']
                # Additional boost for diverse or high-confidence metrics
                if len(metric_contexts) >= 2:
                    score *= 1.1  # 10% boost for multiple metrics
            if context['tool_usage']:
                score *= self.context_multipliers['tool_usage']
            if context['leadership_indicators']:
                score *= self.context_multipliers['leadership']
                
            context_scores.append(score)
            
        # Return highest context score
        return max(context_scores)
        
    def _find_achievement_indicators(self, text: str) -> List[str]:
        """Find achievement-related terms in text"""
        indicators = [
            'improved', 'increased', 'reduced', 'achieved', 'delivered',
            'implemented', 'developed', 'launched', 'led', 'managed',
            'optimized', 'enhanced', 'streamlined', 'transformed'
        ]
        return [word for word in indicators if word in text.lower()]
        

        
    def _find_tool_usage(self, text: str) -> List[str]:
        """Find mentions of tool usage"""
        tool_patterns = [
            r'using\s+\w+',
            r'utilized\s+\w+',
            r'implemented\s+\w+',
            # Common CS/CX tools and platforms
            # CRM and Customer Support
            r'(?:using|utilized|worked with)\s+(?:salesforce|zendesk|intercom|freshdesk|servicenow)',
            r'(?:experience|expertise|proficiency)\s+(?:with|in)\s+(?:hubspot|dynamics|zoho|desk\.com|help ?scout)',
            
            # Ticketing and Issue Tracking
            r'(?:administered|managed|configured)\s+(?:jira|confluence|trello|asana|monday\.com)',
            r'(?:basecamp|notion|clickup|linear|gitlab)\s+(?:administration|configuration|setup)',
            
            # Communication and Collaboration
            r'(?:implemented|deployed|integrated)\s+(?:slack|teams|zoom|webex|ringcentral)',
            r'(?:front|outreach|gong|chorus|aircall)\s+(?:implementation|deployment|integration)',
            
            # Analytics and Reporting
            r'(?:expert|advanced|proficient)\s+(?:tableau|looker|power ?bi|google analytics)',
            r'(?:mixpanel|amplitude|pendo|gainsight|totango)\s+(?:user|administrator)',
            
            # Knowledge Base and Documentation
            r'(?:trained|supported|guided)\s+(?:users|team|clients)\s+(?:on|in)\s+(?:confluence|notion|guru|gitbook|readme\.io)',
            r'(?:provided|delivered)\s+(?:helpjuice|document360|helpscout docs)\s+(?:training|support|guidance)',
            
            # Project Management
            r'(?:managed|coordinated)\s+(?:asana|trello|monday\.com|clickup|basecamp)',
            r'(?:smartsheet|workfront|wrike|teamwork)\s+(?:project|task|workflow)\s+(?:management|coordination)',
            
            # Survey and Feedback
            r'(?:collected|analyzed)\s+(?:feedback|data)\s+(?:using|via)\s+(?:surveymonkey|typeform|qualtrics|delighted)',
            r'(?:wootric|uservoice|canny|productboard)\s+(?:feedback|insight|analysis)'
        ]
        matches = []
        for pattern in tool_patterns:
            matches.extend(re.findall(pattern, text.lower()))
        return len(matches) > 0
        
    def _find_leadership_indicators(self, text: str) -> List[str]:
        """Find leadership-related terms"""
        indicators = [
            'led', 'managed', 'supervised', 'directed', 'coordinated',
            'mentored', 'trained', 'guided', 'spearheaded', 'orchestrated'
        ]
        return [word for word in indicators if word in text.lower()]
        
    def _check_context_type(self, text: str, context_type: str) -> bool:
        """Check for specific context types in text"""
        text = text.lower()
        
        if context_type == 'direct_mention':
            patterns = [
                r'proficient in',
                r'skilled in',
                r'expertise in',
                r'specialized in'
            ]
            return any(re.search(pattern, text) for pattern in patterns)
            
        elif context_type == 'achievement':
            patterns = [
                r'successfully',
                r'improved',
                r'increased',
                r'reduced',
                r'achieved',
                r'delivered'
            ]
            return any(re.search(pattern, text) for pattern in patterns)
            
        elif context_type == 'tool_usage':
            # Common CS/CX tool usage patterns
            patterns = [
                # General usage
                r'(?:using|utilized|worked with|experienced with|proficient in)\s+(?:\w+\s+)*(?:tool|platform|system|software)',
                
                # Administration
                r'(?:administered|managed|configured|maintained)\s+(?:\w+\s+)*(?:tool|platform|system|software)',
                r'(?:tool|platform|system|software)\s+(?:administration|configuration|maintenance)',
                
                # Implementation
                r'(?:implemented|deployed|integrated|installed)\s+(?:\w+\s+)*(?:tool|platform|system|software)',
                r'(?:tool|platform|system|software)\s+(?:implementation|deployment|integration)',
                
                # Expertise
                r'(?:expert|advanced|proficient)\s+(?:\w+\s+)*(?:user|administrator)',
                r'(?:certification|certified|trained)\s+(?:in|on)\s+(?:\w+\s+)*(?:tool|platform|system|software)',
                
                # Training/Support
                r'(?:trained|supported|guided)\s+(?:users|team|clients)\s+(?:on|in)\s+(?:\w+\s+)*(?:tool|platform|system|software)',
                r'(?:provided|delivered)\s+(?:tool|platform|system|software)\s+(?:training|support|guidance)',
                
                # Specific CS/CX tools
                r'(?:crm|helpdesk|ticketing|analytics|reporting|knowledge base|project management)\s+(?:tool|platform|system|software)',
                r'(?:salesforce|zendesk|intercom|freshdesk|servicenow|hubspot|jira|confluence)',
                r'(?:tableau|looker|power bi|google analytics|mixpanel|amplitude)',
                r'(?:slack|teams|zoom|webex|ringcentral|front|outreach|gong)',
                
                # Tool impact
                r'(?:improved|optimized|enhanced)\s+(?:\w+\s+)*(?:using|through|with)\s+(?:\w+\s+)*(?:tool|platform|system|software)',
                r'(?:automation|efficiency|productivity)\s+(?:through|using|with)\s+(?:\w+\s+)*(?:tool|platform|system|software)'
            ]
            return any(re.search(pattern, text.lower()) for pattern in patterns)
            
        elif context_type == 'responsibility':
            patterns = [
                r'responsible for',
                r'managed',
                r'led',
                r'coordinated',
                r'oversaw'
            ]
            return any(re.search(pattern, text) for pattern in patterns)
            
        return False
        
    def _check_false_positive(self, skill: str, text: str) -> bool:
        """Enhanced false positive detection with context awareness"""
        text_lower = text.lower()
        skill_lower = skill.lower()
        
        # Check common false positive patterns
        false_positive_patterns = [
            r'familiarity with\s+' + re.escape(skill_lower),
            r'exposure to\s+' + re.escape(skill_lower),
            r'basic understanding of\s+' + re.escape(skill_lower),
            r'knowledge of\s+' + re.escape(skill_lower),
            r'learn\s+' + re.escape(skill_lower),
            r'studying\s+' + re.escape(skill_lower),
            r'interested in\s+' + re.escape(skill_lower),
            r'would like to\s+' + re.escape(skill_lower)
        ]
        
        if any(re.search(pattern, text_lower) for pattern in false_positive_patterns):
            return True
            
        # Check for weak or aspirational phrases
        weak_phrases = [
            'willing to learn',
            'eager to learn',
            'hope to',
            'looking to',
            'want to',
            'basic level',
            'beginner level',
            'entry level'
        ]
        
        # Find the sentence containing the skill
        skill_sentence = next((s for s in sent_tokenize(text_lower) if skill_lower in s), '')
        if any(phrase in skill_sentence for phrase in weak_phrases):
            return True
            
        # Check skill context requirements
        skill_type = self._determine_skill_type(skill)
        if skill_type in self.skill_confidence_adjustments:
            required_context = self.skill_confidence_adjustments[skill_type]['context_requirement']
            sentences = sent_tokenize(text)
            contexts = self._find_skill_contexts(skill, sentences)
            context_score = self._analyze_skill_context(skill, contexts)
            if context_score < required_context:
                return True
                
        return False
        
    def _find_skill_mentions(self, skill: str, text: str) -> List[int]:
        """Find all mentions of a skill in text"""
        skill = skill.lower()
        text = text.lower()
        mentions = []
        
        start = 0
        while True:
            pos = text.find(skill, start)
            if pos == -1:
                break
            mentions.append(pos)
            start = pos + 1
            
        return mentions
        
    def _get_surrounding_context(self, position: int, text: str, window: int = 100) -> str:
        """Get surrounding context for a position in text"""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end]
        
    def _determine_skill_type(self, skill: str) -> Optional[str]:
        """Determine the type of a skill"""
        skill = skill.lower()
        
        # Define type patterns with more specific coverage
        type_patterns = {
            'customer_service': r'(?:customer|client)\s+(?:service|support|success|experience|satisfaction)',
            'technical_support': r'(?:technical|product)\s+(?:support|troubleshooting|documentation)',
            'account_management': r'(?:account|client|relationship)\s+(?:management|success|retention)',
            'data_analysis': r'(?:data|metrics|analytics)\s+(?:analysis|reporting|tracking)',
            'crm_tools': r'(?:crm|salesforce|zendesk|intercom|freshdesk|servicenow)',
            'process_optimization': r'(?:process|workflow|system)\s+(?:optimization|improvement|efficiency)',
            'stakeholder_management': r'(?:stakeholder|relationship|communication)\s+(?:management|engagement)',
            'project_management': r'(?:project|program|initiative)\s+(?:management|coordination|planning)',
            'leadership': r'(?:team|department|organization)\s+(?:leadership|management|direction)',
            'communication': r'(?:verbal|written|presentation|documentation)\s+(?:skills|communication|ability)'
        }
        
        # Check each type pattern
        matched_types = []
        for skill_type, pattern in type_patterns.items():
            if re.search(pattern, skill):
                matched_types.append((skill_type, len(re.findall(pattern, skill))))
        
        # Return the type with the most matches, or the first one if tied
        if matched_types:
            return max(matched_types, key=lambda x: x[1])[0]
        
        # Fallback to more general patterns
        general_patterns = {
            'customer_service': r'support|service|success',
            'technical_support': r'technical|software|system',
            'account_management': r'account|client|relationship',
            'data_analysis': r'data|analytics|metrics',
            'crm_tools': r'crm|tool|platform',
            'process_optimization': r'process|workflow|efficiency',
            'stakeholder_management': r'stakeholder|engagement|coordination',
            'project_management': r'project|program|initiative',
            'leadership': r'leadership|management|direction',
            'communication': r'communication|presentation|documentation'
        }
        
        for skill_type, pattern in general_patterns.items():
            if re.search(pattern, skill):
                return skill_type
        
        return None
        
    def _calculate_confidence(self, skill: str, context_score: float, contexts: List[Dict]) -> float:
        """Calculate confidence with enhanced context awareness and domain-specific boosts"""
        # Base confidence from context score
        confidence = context_score * self.base_confidence
        
        # Get skill type for domain-specific adjustments
        skill_type = self._determine_skill_type(skill)
        
        # Apply context multipliers with domain awareness
        context_types_found = set()
        for context in contexts:
            context_type = context.get('type', '')
            if not context_type:
                continue
                
            context_types_found.add(context_type)
            
            # Apply multiplier if available
            if context_type in self.context_multipliers:
                multiplier = self.context_multipliers[context_type]
                
                # Boost multiplier for domain-specific contexts
                if skill_type:
                    if (
                        (skill_type == 'customer_service' and context_type in ['customer_interaction', 'direct_mention']) or
                        (skill_type == 'technical_support' and context_type in ['technical_expertise', 'tool_usage']) or
                        (skill_type == 'account_management' and context_type in ['responsibility', 'customer_interaction']) or
                        (skill_type == 'data_analysis' and context_type in ['quantified', 'technical_expertise']) or
                        (skill_type == 'crm_tools' and context_type in ['tool_usage', 'technical_expertise']) or
                        (skill_type == 'process_optimization' and context_type in ['process_improvement', 'quantified']) or
                        (skill_type == 'stakeholder_management' and context_type in ['leadership', 'team_collaboration']) or
                        (skill_type == 'project_management' and context_type in ['leadership', 'responsibility']) or
                        (skill_type == 'leadership' and context_type in ['leadership', 'team_collaboration']) or
                        (skill_type == 'communication' and context_type in ['direct_mention', 'team_collaboration'])
                    ):
                        multiplier *= 1.2  # Additional 20% boost for domain-aligned contexts
                        
                confidence *= multiplier
        
        # Apply skill-specific confidence adjustments
        if skill_type and skill_type in self.skill_confidence_adjustments:
            adjustment = self.skill_confidence_adjustments[skill_type]
            
            # Apply base boost
            confidence *= adjustment['base_boost']
            
            # Check if required contexts are present
            required_score = adjustment['context_requirement']
            if context_score >= required_score:
                confidence *= 1.1  # Additional 10% boost for meeting context requirements
        
        # Boost confidence if multiple relevant contexts found
        relevant_context_count = len(context_types_found)
        if relevant_context_count > 2:
            confidence *= (1.0 + (0.05 * (relevant_context_count - 2)))  # 5% boost per additional context
        
        # Boost for quantified achievements
        has_metrics = any(context.get('quantified_metrics', False) for context in contexts)
        if has_metrics:
            confidence *= 1.2  # 20% boost for quantified achievements
        
        # Evidence-based boosts
        context_text = ' '.join(str(c.get('text', '')) for c in contexts)
        if self._has_implementation_evidence(skill, context_text):
            confidence *= 1.15  # 15% boost for implementation evidence
            
        if self._has_team_impact(skill, context_text):
            confidence *= 1.1  # 10% boost for team impact
        
        # Normalize confidence
        confidence = min(confidence, 1.0)
        
        return confidence
        
    def _apply_domain_boosts(self, skills: List[Dict], text: str) -> List[Dict]:
        """Apply domain-specific confidence boosts"""
        boosted = []
        text_lower = text.lower()
        
        for skill in skills:
            skill_info = skill.copy()
            
            # Boost for CS/CX specific terms
            if any(term in skill['skill'].lower() for term in ['customer', 'client', 'success', 'experience']):
                skill_info['confidence'] *= 1.2
            
            # Boost for technical skills with implementation evidence
            if self._has_implementation_evidence(skill['skill'], text_lower):
                skill_info['confidence'] *= 1.15
            
            # Boost for skills with metrics
            if any(context.get('quantified_metrics') for context in skill['contexts']):
                skill_info['confidence'] *= 1.1
            
            # Boost for leadership skills with team impact
            if self._has_team_impact(skill['skill'], text_lower):
                skill_info['confidence'] *= 1.15
            
            boosted.append(skill_info)
            
        return boosted
        
    def _has_implementation_evidence(self, skill: str, text: str) -> bool:
        """Check for evidence of skill implementation"""
        implementation_patterns = [
            r'implemented\s+\w+\s+using\s+' + re.escape(skill.lower()),
            r'developed\s+\w+\s+with\s+' + re.escape(skill.lower()),
            r'built\s+\w+\s+using\s+' + re.escape(skill.lower()),
            r'deployed\s+\w+\s+with\s+' + re.escape(skill.lower())
        ]
        return any(re.search(pattern, text) for pattern in implementation_patterns)
        
    def _has_team_impact(self, skill: str, text: str) -> bool:
        """Check for evidence of team/organizational impact"""
        impact_patterns = [
            r'led\s+team\s+\w+\s+' + re.escape(skill.lower()),
            r'managed\s+\w+\s+team\s+' + re.escape(skill.lower()),
            r'trained\s+\w+\s+on\s+' + re.escape(skill.lower()),
            r'improved\s+team\s+\w+\s+using\s+' + re.escape(skill.lower())
        ]
        return any(re.search(pattern, text) for pattern in impact_patterns)
        
    def _custom_analyzer(self, text: str) -> List[str]:
        """Custom text analyzer for TF-IDF"""
        words = word_tokenize(text.lower())
        return [self.lemmatizer.lemmatize(word) for word in words 
                if word not in self.stop_words and len(word) > 2]
                
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Basic cleaning
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize common variations
        text = text.replace('&', 'and')
        text = re.sub(r'(\d+)([kK])\b', r'\1000', text)  # Convert 10k to 10000
        text = re.sub(r'(\d+)([mM])\b', r'\1000000', text)  # Convert 1m to 1000000
        
        return text.strip()
        
    def _validate_skill_coherence(self, skill: str) -> bool:
        """Validate skill coherence with domain knowledge"""
        skill_lower = skill.lower()
        
        # CS/CX domain terms
        domain_terms = {
            # Core CS/CX terms
            'customer', 'client', 'service', 'support', 'success', 'experience',
            'account', 'relationship', 'satisfaction', 'retention', 'engagement',
            
            # Technical terms
            'technical', 'troubleshooting', 'documentation', 'knowledge',
            'crm', 'helpdesk', 'ticketing', 'salesforce', 'zendesk', 'jira',
            
            # Process terms
            'onboarding', 'escalation', 'resolution', 'sla', 'kpi', 'metrics',
            'workflow', 'process', 'protocol', 'procedure', 'optimization',
            
            # Communication terms
            'communication', 'presentation', 'documentation', 'stakeholder',
            'training', 'coaching', 'mentoring', 'collaboration',
            
            # Analytical terms
            'analysis', 'reporting', 'tracking', 'monitoring', 'analytics',
            'insights', 'performance', 'metrics', 'data', 'trends',
            
            # Project terms
            'project', 'program', 'initiative', 'implementation', 'deployment',
            'coordination', 'management', 'planning', 'execution',
            
            # Soft skills
            'leadership', 'teamwork', 'collaboration', 'empathy', 'patience',
            'adaptability', 'flexibility', 'organization', 'prioritization'
        }
        
        # Check if skill contains domain terms
        if any(term in skill_lower for term in domain_terms):
            return True
            
        # Check multi-word skills against patterns
        if len(skill.split()) > 1:
            patterns = [
                # Customer interaction patterns
                r'(?:customer|client)\s+(?:\w+\s+)*(?:service|support|success|experience)',
                r'(?:account|relationship)\s+(?:management|executive|coordinator)',
                
                # Technical patterns
                r'(?:technical|product)\s+(?:support|specialist|consultant)',
                r'(?:crm|helpdesk|ticketing)\s+(?:system|software|administration)',
                
                # Process patterns
                r'(?:process|workflow|system)\s+(?:optimization|improvement|management)',
                r'(?:quality|performance)\s+(?:assurance|monitoring|management)',
                
                # Communication patterns
                r'(?:stakeholder|team|cross-functional)\s+(?:communication|collaboration)',
                r'(?:verbal|written)\s+communication\s+(?:skills|ability)',
                
                # Project patterns
                r'(?:project|program|initiative)\s+(?:management|coordination)',
                r'(?:change|release)\s+(?:management|planning|execution)',
                
                # Analytical patterns
                r'(?:data|metrics|performance)\s+(?:analysis|reporting|tracking)',
                r'(?:business|customer)\s+(?:intelligence|analytics|insights)',
                
                # General skill patterns
                r'\w+\s+(?:management|support|service|analysis|skills)',
                r'(?:proficiency|expertise)\s+(?:in|with)\s+\w+',
                r'\w+\s+(?:optimization|improvement|development)'
            ]
            
            return any(re.match(pattern, skill_lower) for pattern in patterns)
            
        return False
        
    def _check_domain_presence(self, skill: str) -> bool:
        """Check if a single word skill has strong domain presence"""
        # Define core domain terms
        core_terms = {
            'customer', 'client', 'account', 'service', 'support',
            'success', 'experience', 'management', 'analysis', 'data',
            'technical', 'product', 'project', 'strategy', 'business'
        }
        
        # Check direct match
        if skill in core_terms:
            return True
            
        # Check partial matches
        return any(term in skill or skill in term for term in core_terms)
        
    def _check_skill_pattern(self, skill: str) -> bool:
        """Check if a multi-word skill follows valid patterns"""
        # Define valid patterns
        patterns = [
            (r'customer|client', r'success|experience|support|service'),
            (r'account|portfolio', r'management|strategy|growth'),
            (r'data|analytics', r'analysis|reporting|visualization'),
            (r'technical|solution', r'implementation|support|architecture'),
            (r'project|program', r'management|coordination|delivery'),
            (r'business|revenue', r'development|strategy|growth')
        ]
        
        # Check against patterns
        return any(re.search(f"\\b({p1})\\s+({p2})\\b", skill) for p1, p2 in patterns)
        
    def match(self, source_skills: Set[str], target_skills: Set[str]) -> float:
        """Match skills using semantic similarity with enhanced precision"""
        if not source_skills or not target_skills:
            return 0.0
            
        # Calculate similarities between all pairs
        similarities = []
        for source in source_skills:
            for target in target_skills:
                similarity = self._calculate_skill_similarity(source, target)
                similarities.append(similarity)
                
        # Return average of top matches
        top_k = min(len(similarities), min(len(source_skills), len(target_skills)))
        return sum(sorted(similarities, reverse=True)[:top_k]) / top_k
        
    def get_skill_weight(self, skill: str) -> float:
        """Get weight based on enhanced skill importance"""
        # Get context-independent base weight
        base_weight = 1.0
        
        # Apply skill-specific adjustments
        skill_type = self._determine_skill_type(skill)
        if skill_type in self.skill_confidence_adjustments:
            base_weight *= self.skill_confidence_adjustments[skill_type]['base_boost']
            
        # Scale weight to be between 0.5 and 3.0
        scaled_weight = 0.5 + (base_weight * 2.5)
        return min(3.0, max(0.5, scaled_weight))
        
    def _calculate_skill_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate semantic similarity between two skills"""
        # Exact match
        if skill1.lower() == skill2.lower():
            return 1.0
            
        # Check synonyms
        if self._are_synonyms(skill1, skill2):
            return 0.9
            
        # Calculate TF-IDF similarity
        vec1 = self._get_text_vector(skill1)
        vec2 = self._get_text_vector(skill2)
        
        similarity = np.dot(vec1, vec2.T)[0, 0]
        return float(similarity)
        
    def _are_synonyms(self, skill1: str, skill2: str) -> bool:
        """Check if two skills are synonyms"""
        skill1, skill2 = skill1.lower(), skill2.lower()
        
        # Check direct synonyms
        for base_skill, synonyms in self.synonyms.items():
            if skill1 in synonyms and skill2 in synonyms:
                return True
                
        return False
