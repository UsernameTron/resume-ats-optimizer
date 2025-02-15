from typing import Set, Dict, List, Tuple, Optional
from .base_matcher import BaseMatcher
from .enhanced_skill_matcher import EnhancedSkillMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import joblib
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

class SemanticMatcher(BaseMatcher):
    """Enhanced semantic-based skill matcher for CS/CX domain"""
    
    def __init__(self, cache_dir: str = "cache"):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize CS/CX-specific components
        self.lemmatizer = WordNetLemmatizer()
        
        # CS/CX domain stopwords
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update({
            'experience', 'skill', 'skills', 'year', 'years',
            'knowledge', 'understanding', 'ability', 'proficiency'
        })
        
        # CS/CX domain weights
        self.domain_weights = {
            # Core terms
            'customer': 2.5,
            'client': 2.5,  # Equal weight for client/customer
            'success': 2.0,
            'experience': 2.0,
            'account': 2.0,
            'management': 2.0,
            
            # Key metrics
            'satisfaction': 1.8,
            'retention': 1.8,
            'churn': 1.8,
            'nps': 1.8,
            'csat': 1.8,
            'mrr': 1.8,
            
            # Important activities
            'relationship': 1.5,
            'stakeholder': 1.5,
            'engagement': 1.5,
            'adoption': 1.5,
            'implementation': 1.5,
            'onboarding': 1.5
        }
        
        # Common synonyms in CS/CX domain
        self.synonyms = {
            # Tools and platforms
            'salesforce': {'sfdc', 'sales force', 'sales-force', 'salesforce crm', 'sales cloud'},
            'gainsight': {'gain sight', 'gain-sight', 'gainsight cs', 'gainsight cx'},
            'zendesk': {'zen desk', 'zen-desk', 'zendesk support', 'zendesk chat'},
            'intercom': {'intercom chat', 'intercom messenger'},
            'hubspot': {'hub spot', 'hubspot crm', 'hubspot service'},
            
            # Core terms
            'customer success': {'cs', 'csm', 'customer success management', 'client success'},
            'customer experience': {'cx', 'client experience', 'customer exp'},
            'account management': {'client management', 'relationship management', 'account relationships'},
            
            # Metrics
            'churn': {'attrition', 'turnover', 'churn rate', 'customer churn'},
            'csat': {'customer satisfaction', 'satisfaction score', 'customer satisfaction score'},
            'nps': {'net promoter score', 'net-promoter-score', 'promoter score'},
            'mrr': {'monthly recurring revenue', 'recurring revenue'},
            'customer health': {'health score', 'account health', 'client health'},
            
            # Processes
            'qbr': {'quarterly business review', 'business review', 'quarterly review'},
            'implementation': {'deployment', 'rollout', 'setup', 'onboarding setup'},
            'onboarding': {'customer onboarding', 'client onboarding', 'user onboarding', 'enablement'},
            'adoption': {'product adoption', 'solution adoption', 'platform adoption'},
            'retention': {'customer retention', 'client retention', 'account retention'}
        }
        
        # Initialize vectorizer with CS/CX optimizations
        self.vectorizer = TfidfVectorizer(
            analyzer=self._custom_analyzer,
            ngram_range=(1, 3),
            max_features=15000,
            min_df=0.0,  # Allow all terms since we have a small corpus
            max_df=1.0,  # Allow all terms since we have a small corpus
            sublinear_tf=True
        )
        
        # Initialize caches
        self.vectors_cache = {}
        self.memory_cache = {}
        self.load_cached_vectors()
        
        # Maximum number of skills to extract
        self.max_skills = 30  # Reasonable default for CS/CX roles
        
        # Fit vectorizer with CS/CX vocabulary
        self._fit_vectorizer()
        
    def _fit_vectorizer(self):
        """Fit vectorizer with CS/CX-specific vocabulary"""
        # Core CS/CX skills
        core_skills = [
            'customer success', 'customer experience', 'account management',
            'client success', 'client experience', 'customer satisfaction',
            'customer support', 'customer service', 'customer engagement',
            'customer retention', 'churn reduction', 'nps', 'csat'
        ]
        
        # Tools and platforms
        tools = [
            'salesforce', 'gainsight', 'zendesk', 'intercom', 'hubspot',
            'freshdesk', 'totango', 'churnzero', 'vitally', 'planhat'
        ]
        
        # Processes and methodologies
        processes = [
            'customer journey', 'customer lifecycle', 'onboarding',
            'implementation', 'adoption', 'escalation management',
            'relationship building', 'stakeholder management'
        ]
        
        # Business metrics
        metrics = [
            'revenue retention', 'upsell', 'cross-sell', 'expansion',
            'customer health', 'quarterly business review', 'qbr',
            'customer advocacy', 'voice of customer'
        ]
        
        # Analytics and insights
        analytics = [
            'customer feedback', 'customer insights', 'customer analytics',
            'customer metrics', 'customer segmentation', 'customer communication'
        ]
        
        # Training and enablement
        enablement = [
            'customer education', 'customer training', 'product adoption',
            'customer success metrics', 'customer success strategy',
            'customer success operations', 'customer success enablement'
        ]
        
        # Create multiple documents for better term frequency distribution
        documents = [
            ' '.join(core_skills),
            ' '.join(tools),
            ' '.join(processes),
            ' '.join(metrics),
            ' '.join(analytics),
            ' '.join(enablement)
        ]
        
        # Fit vectorizer with domain vocabulary
        self.vectorizer.fit(documents)
        
    def load_cached_vectors(self):
        """Load cached TF-IDF vectors with memory mapping"""
        cache_file = self.cache_dir / "tfidf_vectors.joblib"
        if cache_file.exists():
            try:
                # Use memory mapping for faster loading
                self.vectors_cache = joblib.load(cache_file, mmap_mode='r')
                # Initialize in-memory cache for frequently accessed items
                self.memory_cache = {}
                print(f"Loaded {len(self.vectors_cache)} cached vectors")
            except Exception as e:
                print(f"Failed to load vectors cache: {str(e)}")
                self.vectors_cache = {}
                self.memory_cache = {}
    
    def save_vectors_cache(self):
        """Save TF-IDF vectors cache with compression"""
        cache_file = self.cache_dir / "tfidf_vectors.joblib"
        try:
            # Use compression for smaller file size, faster I/O
            joblib.dump(self.vectors_cache, cache_file, compress=3)
            # Update memory cache with new entries
            self.memory_cache.update(self.vectors_cache)
            print(f"Saved {len(self.vectors_cache)} vectors to cache")
        except Exception as e:
            print(f"Failed to save vectors cache: {str(e)}")
    
    def _custom_analyzer(self, text: str) -> List[str]:
        """Custom analyzer with CS/CX domain optimization"""
        # Normalize text
        text = text.lower()
        
        # First replace multi-word synonyms
        for term, synonyms in self.synonyms.items():
            if ' ' in term:
                for syn in synonyms:
                    if syn in text:
                        text = text.replace(syn, term)
        
        # Then replace single-word synonyms
        for term, synonyms in self.synonyms.items():
            if ' ' not in term:
                for syn in synonyms:
                    if syn in text:
                        text = text.replace(syn, term)
        
        # Apply CS/CX term normalization
        text = self._normalize_cs_terms(text)
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove stopwords and non-alphanumeric tokens
        tokens = [token for token in tokens if token not in self.stop_words 
                 and (token.isalnum() or token in self.synonyms)]
        
        # Generate n-grams for domain terms
        ngrams = []
        for i in range(len(tokens)-1):
            bigram = ' '.join(tokens[i:i+2])
            if bigram in self.synonyms or any(term in bigram for term in self.domain_weights):
                ngrams.append(bigram)
        
        # Add weighted tokens
        weighted_tokens = []
        for token in tokens + ngrams:
            weight = self.domain_weights.get(token, 1.0)
            weighted_tokens.extend([token] * int(weight * 10))
        
        return weighted_tokens
    
    def _get_text_vector(self, text: str) -> np.ndarray:
        """Get TF-IDF vector for text with optimized caching"""
        # Normalize CS/CX variations
        text = self._normalize_cs_terms(text)
        
        # Convert vector to bytes for caching
        cache_key = text.encode('utf-8')
        
        # Check in-memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
            
        # Check disk cache next
        if cache_key in self.vectors_cache:
            vector = self.vectors_cache[cache_key]
            # Cache frequently accessed items in memory
            self.memory_cache[cache_key] = vector
            return vector
            
        # Transform text to TF-IDF vector
        vector = self.vectorizer.transform([text]).toarray()[0]
        
        # Cache the vector in both caches
        self.vectors_cache[cache_key] = vector
        self.memory_cache[cache_key] = vector
        
        # Save cache if it reaches certain size
        if len(self.vectors_cache) % 100 == 0:
            self.save_vectors_cache()
            
        return vector
    
    def _normalize_cs_terms(self, text: str) -> str:
        """Normalize common CS/CX terms and variations"""
        # Verb-noun variations
        verb_noun_pairs = {
            # Customer Success Core
            'reduce churn': 'churn reduction',
            'retain customers': 'customer retention',
            'manage accounts': 'account management',
            'support customers': 'customer support',
            'engage clients': 'client engagement',
            'build relationships': 'relationship building',
            'implement solutions': 'solution implementation',
            'onboard customers': 'customer onboarding',
            'adopt product': 'product adoption',
            'satisfy customers': 'customer satisfaction',
            'drive retention': 'retention management',
            'improve satisfaction': 'satisfaction improvement',
            'track metrics': 'metrics tracking',
            'analyze performance': 'performance analysis',
            'manage stakeholders': 'stakeholder management',
            'lead reviews': 'review leadership',
            'optimize processes': 'process optimization',
            
            # Expansion and Growth
            'expand accounts': 'account expansion',
            'grow revenue': 'revenue growth',
            'increase adoption': 'adoption increase',
            'scale operations': 'operational scaling',
            'develop strategies': 'strategy development',
            'execute plans': 'plan execution',
            
            # Risk and Health
            'monitor health': 'health monitoring',
            'assess risks': 'risk assessment',
            'identify opportunities': 'opportunity identification',
            'forecast renewals': 'renewal forecasting',
            'predict churn': 'churn prediction',
            'measure satisfaction': 'satisfaction measurement',
            
            # Communication and Leadership
            'present reviews': 'review presentation',
            'facilitate meetings': 'meeting facilitation',
            'coordinate teams': 'team coordination',
            'mentor colleagues': 'colleague mentorship',
            'coach teams': 'team coaching',
            'resolve escalations': 'escalation resolution',
            
            # Technical and Tools
            'configure systems': 'system configuration',
            'customize platforms': 'platform customization',
            'integrate tools': 'tool integration',
            'automate processes': 'process automation',
            'generate reports': 'report generation',
            'analyze data': 'data analysis'
        }
        
        # Standard term replacements
        replacements = {
            # Core roles and functions
            'customer success': 'customersuccess',
            'customer experience': 'customerexperience',
            'account management': 'accountmanagement',
            'client success': 'customersuccess',
            'client experience': 'customerexperience',
            'customer service': 'customerservice',
            'portfolio management': 'portfoliomanagement',
            'relationship management': 'relationshipmanagement',
            'success management': 'successmanagement',
            
            # Metrics and KPIs
            'customer satisfaction': 'customersatisfaction',
            'customer retention': 'customerretention',
            'churn rate': 'churn',
            'net promoter score': 'nps',
            'satisfaction score': 'csat',
            'quarterly business review': 'qbr',
            'monthly recurring revenue': 'mrr',
            'annual recurring revenue': 'arr',
            'customer health score': 'customerhealth',
            'time to value': 'timetovalue',
            'time to resolution': 'timetoresolution',
            'first response time': 'firstresponsetime',
            'customer effort score': 'customereffort',
            
            # Processes and Activities
            'customer onboarding': 'customeronboarding',
            'product adoption': 'productadoption',
            'account expansion': 'accountexpansion',
            'business review': 'businessreview',
            'stakeholder management': 'stakeholdermanagement',
            'escalation management': 'escalationmanagement',
            'renewal management': 'renewalmanagement',
            'project management': 'projectmanagement',
            'change management': 'changemanagement',
            'risk management': 'riskmanagement',
            
            # Tools and Platforms
            'sales force': 'salesforce',
            'gain sight': 'gainsight',
            'zen desk': 'zendesk',
            'hub spot': 'hubspot',
            'fresh desk': 'freshdesk',
            'service now': 'servicenow',
            'jira service desk': 'jiraservicedesk',
            'customer success platform': 'customersuccessplatform',
            'crm system': 'crmsystem',
            'help desk': 'helpdesk',
            
            # Methodologies and Frameworks
            'agile methodology': 'agilemethodology',
            'customer journey mapping': 'customerjourneymap',
            'voice of customer': 'voiceofcustomer',
            'success planning': 'successplanning',
            'strategic planning': 'strategicplanning',
            'account planning': 'accountplanning'
        }
        
        text = text.lower()
        
        # Handle verb-noun variations first
        for verb_form, noun_form in verb_noun_pairs.items():
            if verb_form in text:
                text = text.replace(verb_form, noun_form)
        
        # Then apply standard replacements
        for original, replacement in replacements.items():
            text = text.replace(original, replacement)
        
        return text
    
    def _normalize_skill(self, skill: str) -> str:
        """Normalize a skill by applying enhanced CS/CX specific handling"""
        # Convert to lowercase and strip whitespace
        skill = skill.lower().strip()
        
        # Handle common metric and term abbreviations
        term_mappings = {
            # Metrics
            'nps': 'net promoter score',
            'csat': 'customer satisfaction',
            'ces': 'customer effort score',
            'mrr': 'monthly recurring revenue',
            'arr': 'annual recurring revenue',
            'ltv': 'lifetime value',
            'clv': 'customer lifetime value',
            'cac': 'customer acquisition cost',
            'qbr': 'quarterly business review',
            'roi': 'return on investment',
            'ttv': 'time to value',
            'frt': 'first response time',
            'mttr': 'mean time to resolution',
            'sla': 'service level agreement',
            
            # Roles and Functions
            'cs': 'customer success',
            'cx': 'customer experience',
            'csm': 'customer success manager',
            'tam': 'technical account manager',
            'ae': 'account executive',
            'am': 'account manager',
            'sa': 'solutions architect',
            
            # Tools and Platforms
            'sfdc': 'salesforce',
            'sf': 'salesforce',
            'gs': 'gainsight',
            'zd': 'zendesk',
            'hs': 'hubspot',
            'ic': 'intercom'
        }
        
        # Replace abbreviated terms with full forms
        words = skill.split()
        normalized_words = []
        i = 0
        while i < len(words):
            # Try to match 2-word abbreviations first
            if i < len(words) - 1:
                two_word = words[i] + ' ' + words[i+1]
                if two_word in term_mappings:
                    normalized_words.extend(term_mappings[two_word].split())
                    i += 2
                    continue
            
            # Then try single word matches
            if words[i] in term_mappings:
                normalized_words.extend(term_mappings[words[i]].split())
            else:
                normalized_words.append(words[i])
            i += 1
        
        skill = ' '.join(normalized_words)
        
        # Handle metric variations
        metric_variations = {
            'churn rate': 'churn',
            'churn percentage': 'churn',
            'customer churn': 'churn',
            'health score': 'customer health',
            'customer health score': 'customer health',
            'health index': 'customer health',
            'satisfaction score': 'customer satisfaction',
            'satisfaction rating': 'customer satisfaction',
            'promoter score': 'net promoter score'
        }
        
        # Apply metric variations
        for variation, standard in metric_variations.items():
            if variation in skill:
                skill = skill.replace(variation, standard)
        
        # Remove common prefixes that don't change meaning
        prefixes = [
            'strong', 'proven', 'demonstrated', 'excellent', 'advanced',
            'extensive', 'deep', 'solid', 'comprehensive', 'hands-on'
        ]
        for prefix in prefixes:
            if skill.startswith(prefix + ' '):
                skill = skill[len(prefix) + 1:]
        
        # Remove common suffixes that don't change meaning
        suffixes = [
            'skills', 'skill', 'abilities', 'ability', 'expertise',
            'experience', 'knowledge', 'proficiency', 'capabilities'
        ]
        for suffix in suffixes:
            if skill.endswith(' ' + suffix):
                skill = skill[:-len(suffix) - 1]
        
        # Check for direct synonyms first
        for term, synonyms in self.synonyms.items():
            if skill in synonyms:
                return term
        
        # Handle verb-noun variations
        verb_noun_map = {
            'reduce': 'reduction',
            'manage': 'management',
            'support': 'support',
            'engage': 'engagement',
            'implement': 'implementation',
            'onboard': 'onboarding',
            'adopt': 'adoption',
            'retain': 'retention',
            'optimize': 'optimization'
        }
        
        words = skill.split()
        for i, word in enumerate(words):
            if word in verb_noun_map:
                words[i] = verb_noun_map[word]
                # Remove 'in' or 'with' if they follow the verb
                if i + 1 < len(words) and words[i + 1] in {'in', 'with'}:
                    words.pop(i + 1)
        
        skill = ' '.join(words)
        
        # Check for partial matches after normalization
        for term, synonyms in self.synonyms.items():
            term_words = set(term.split())
            skill_words = set(skill.split())
            if len(skill_words & term_words) >= len(term_words) / 2:
                return term
        
        # Apply standard CS/CX term normalization
        return self._normalize_cs_terms(skill)
    
    def match(self, source_skills: Set[str], target_skills: Set[str]) -> float:
        """Match skills using semantic similarity with CS/CX optimization"""
        if not source_skills or not target_skills:
            return 0.0
            
        # Create cache key for this comparison
        source_key = ','.join(sorted(source_skills)).encode('utf-8')
        target_key = ','.join(sorted(target_skills)).encode('utf-8')
        match_key = source_key + b'|' + target_key
        
        # Check memory cache first
        if match_key in self.memory_cache:
            return self.memory_cache[match_key]
        
        # Normalize skills
        source_normalized = {self._normalize_skill(skill) for skill in source_skills}
        target_normalized = {self._normalize_skill(skill) for skill in target_skills}
        
        # Calculate exact match score
        exact_matches = source_normalized.intersection(target_normalized)
        exact_score = len(exact_matches) / max(len(source_normalized), len(target_normalized)) if exact_matches else 0.0
        
        # Convert skills to space-separated strings for semantic matching
        source_text = " ".join(source_normalized)
        target_text = " ".join(target_normalized)
        
        # Get vectors
        source_vector = self._get_text_vector(source_text)
        target_vector = self._get_text_vector(target_text)
        
        # Calculate semantic similarity
        semantic_score = cosine_similarity(
            source_vector.reshape(1, -1),
            target_vector.reshape(1, -1)
        )[0][0]
        
        # Calculate domain-specific boosts
        source_tokens = set(word_tokenize(source_text))
        target_tokens = set(word_tokenize(target_text))
        shared_tokens = source_tokens & target_tokens
        
        # Core CS/CX terms (highest weight)
        cs_terms = {
            'customer': 0.5, 'success': 0.5, 'experience': 0.45,
            'account': 0.45, 'client': 0.45, 'management': 0.4
        }
        cs_boost = sum(cs_terms[term] for term in shared_tokens if term in cs_terms)
        
        # Tools and platforms
        tool_terms = {
            'salesforce': 0.4, 'gainsight': 0.4, 'zendesk': 0.35,
            'hubspot': 0.35, 'intercom': 0.35
        }
        tool_boost = sum(tool_terms[term] for term in shared_tokens if term in tool_terms)
        
        # Key metrics
        metric_terms = {
            'nps': 0.35, 'csat': 0.35, 'churn': 0.35,
            'retention': 0.35, 'mrr': 0.3, 'revenue': 0.3
        }
        metric_boost = sum(metric_terms[term] for term in shared_tokens if term in metric_terms)
        
        # Processes
        process_terms = {
            'onboarding': 0.3, 'implementation': 0.3,
            'adoption': 0.3, 'qbr': 0.25, 'strategy': 0.25
        }
        process_boost = sum(process_terms[term] for term in shared_tokens if term in process_terms)
        
        # Calculate total boost with diminishing returns
        total_boost = cs_boost + tool_boost + metric_boost + process_boost
        boost_factor = total_boost / (1 + total_boost)  # Diminishing returns
        
        # Apply boosts to both scores
        exact_score = min(1.0, exact_score * (1 + boost_factor))
        semantic_score = min(1.0, semantic_score * (1 + (boost_factor * 0.8)))  # Less boost for semantic
        
        # Weight exact matches more heavily for domain-specific skills
        if total_boost > 0:
            final_score = (0.7 * exact_score) + (0.3 * semantic_score)
        else:
            final_score = (0.5 * exact_score) + (0.5 * semantic_score)
            
        # Cache the final score
        self.memory_cache[match_key] = float(final_score)
        
        return float(final_score)
        
    def _calculate_domain_boost(self, term: str) -> float:
        """Calculate domain-specific boost for a term"""
        # Check domain weights
        if term in self.domain_weights:
            return self.domain_weights[term]
            
        # Check synonyms
        for domain_term, synonyms in self.synonyms.items():
            if term in synonyms:
                base_term = domain_term.split()[0]  # Get first word
                if base_term in self.domain_weights:
                    return self.domain_weights[base_term] * 0.9  # Slightly lower weight for synonyms
                    
        # Default boost
        return 1.0
    
    def extract_skills(self, text: str) -> Set[str]:
        """Extract skills using enhanced matcher with improved precision and recall"""
        # Use enhanced skill matcher
        enhanced_matcher = EnhancedSkillMatcher(self.cache_dir)
        return enhanced_matcher.extract_skills(text)
        
        # Split text into sentences for contextual analysis
        sentences = nltk.sent_tokenize(text)
        
        # Calculate base TF-IDF scores
        text_vector = self._get_text_vector(text)
        if text_vector is not None:
            # Handle both sparse and dense arrays
            if hasattr(text_vector, 'toarray'):
                vector_array = text_vector.toarray()[0]
            else:
                vector_array = text_vector
                
            for word, score in zip(self.vectorizer.get_feature_names_out(), vector_array):
                if score > 0:
                    base_scores[word] = score
        
        # Calculate contextual scores for each sentence
        sentence_scores = {}
        for sentence in sentences:
            sentence_scores[sentence] = self._calculate_context_score(sentence)
        
        # Calculate contextual scores for each word
        context_scores = {}
        for sentence in sentences:
            context_score = sentence_scores[sentence]
            for word in sentence.split():
                if word in base_scores:
                    if word not in context_scores:
                        context_scores[word] = []
                    context_scores[word].append(context_score)
        
        # Combine base scores with context scores
        final_scores = {}
        for word, score in base_scores.items():
            if word in context_scores:
                avg_context = sum(context_scores[word]) / len(context_scores[word])
                final_scores[word] = score * avg_context
            else:
                final_scores[word] = score
        
        # Filter skills based on scores and domain validation
        if final_scores:
            score_threshold = np.percentile(list(final_scores.values()), 75)  # Dynamic threshold
            for word, score in final_scores.items():
                if score >= score_threshold and self._validate_skill_coherence(word):
                    skills.add(word)
        
        # Extract skills from quantitative achievements
        achievement_patterns = [
            # Improvements and increases
            (r'(?:increased|improved|grew|boosted|enhanced)\s+([\w\s]+)\s+by\s+\d+%?',
             lambda m: f"{m.group(1).strip()} optimization"),
            
            # Reductions
            (r'(?:reduced|decreased|lowered|minimized)\s+([\w\s]+)\s+by\s+\d+%?',
             lambda m: f"{m.group(1).strip()} reduction"),
            
            # Maintenance and targets
            (r'(?:maintained|achieved|reached|sustained)\s+\d+%?\s+([\w\s]+)',
             lambda m: m.group(1).strip()),
            
            # Success metrics
            (r'\d+%\s+(?:increase|improvement|growth|reduction|decrease)\s+in\s+([\w\s]+)',
             lambda m: f"{m.group(1).strip()} optimization")
        ]
        
        for pattern, transform in achievement_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                skill = transform(match)
                if skill:
                    skills.add(skill)
        
        # Extract skills from experience statements
        experience_patterns = [
            # Years of experience
            (r'\d+\s+years?\s+(?:of\s+)?experience\s+(?:in|with)\s+([\w\s]+)',
             lambda m: m.group(1).strip()),
            
            # Expertise levels
            (r'(?:expert|proficient|specialized|experienced|skilled)\s+in\s+([\w\s]+)',
             lambda m: m.group(1).strip()),
            
            # Track record
            (r'(?:proven|demonstrated|successful)\s+(?:track\s+record|history|experience)\s+(?:in|with)\s+([\w\s]+)',
             lambda m: m.group(1).strip()),
            
            # Leadership and ownership
            (r'(?:led|managed|directed|spearheaded)\s+([\w\s]+)\s+(?:team|initiative|program|project)',
             lambda m: f"{m.group(1).strip()} leadership")
        ]
        
        for pattern, transform in experience_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                skill = transform(match)
                if skill:
                    skills.add(skill)
        
        # Enhanced normalization with context preservation
        text = self._normalize_cs_terms(text)
        
        # First normalize metric variations to standardize format
        metric_replacements = {
            r'\b(?:nps|net\s+promoter)\s+(?:score|rating)s?\b': 'nps',
            r'\b(?:csat|customer\s+satisfaction)\s+(?:score|rating)s?\b': 'csat',
            r'\b(?:mrr|monthly\s+recurring\s+revenue)\b': 'mrr',
            r'\b(?:arr|annual\s+recurring\s+revenue)\b': 'arr',
            r'\b(?:logo|customer|client)\s+churn\s+(?:rate|percentage)?\b': 'churn rate',
            r'\b(?:customer|client)\s+retention\s+(?:rate|percentage)?\b': 'retention rate'
        }
        
        for pattern, replacement in metric_replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Now remove numbers while preserving standardized metrics
        metrics = {'nps', 'csat', 'mrr', 'arr'}
        words = text.split()
        filtered_words = []
        
        for i, word in enumerate(words):
            # Keep word if it's a metric or doesn't contain numbers
            if word.lower() in metrics or not any(c.isdigit() for c in word):
                filtered_words.append(word)
        
        text = ' '.join(filtered_words)
        
        # Handle role variations
        role_patterns = [
            # Core roles with seniority
            r'\b(?:senior|lead|principal|head of|director of|vp of)?\s*(?:customer|client)\s+(?:success|experience)\s+(?:manager|lead|specialist|professional|expert|consultant|representative|advocate|champion)\b',
            
            # Account/Portfolio roles
            r'\b(?:strategic|enterprise|senior|lead|principal)?\s*(?:account|portfolio|relationship)\s+(?:manager|executive|director|lead)\b',
            
            # Operations/Program roles
            r'\b(?:customer|client|cs|cx)\s+(?:operations|programs|enablement|strategy)\s+(?:manager|lead|specialist|analyst)\b'
        ]
        
        for pattern in role_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                role = match.group().strip()
                base_role = re.sub(r'\b(?:senior|lead|principal|head of|director of|vp of)\s+', '', role)
                skills.add(base_role)
                if 'senior' in role or 'lead' in role or 'principal' in role:
                    skills.add('leadership')
                if 'enterprise' in role or 'strategic' in role:
                    skills.add('enterprise account management')
        
        # Handle action-oriented achievements
        
        # Return normalized skills
        return {self._normalize_skill(skill) for skill in skills}
        achievement_patterns = [
            # Metrics improvements
            (r'\b(improve|increase|enhance|grow|boost)\s+(retention|satisfaction|adoption|engagement|revenue|expansion)\b',
             lambda m: f"{m.group(2)} optimization"),
            
            # Reductions
            (r'\b(reduce|decrease|lower|minimize)\s+(churn|costs|time|friction)\b',
             lambda m: f"{m.group(2)} reduction"),
            
            # Leadership
            (r'\b(lead|manage|oversee|direct|supervise)\s+(team|initiative|program|project|portfolio)\b',
             lambda m: 'leadership'),
            
            # Implementation
            (r'\b(implement|deploy|launch|roll\s*out|integrate)\s+(solution|platform|tool|system|process)\b',
             lambda m: 'implementation'),
            
            # Analysis
            (r'\b(analyze|monitor|track|measure|assess)\s+(performance|metrics|kpis|health|usage)\b',
             lambda m: 'analytics')
        ]
        
        for pattern, transform in achievement_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                skill = transform(match)
                if skill:
                    skills.add(skill)
        
        # First, extract multi-word skills using regex patterns
        patterns = [
            # Customer Success Core
            r'\b(?:customer|client)\s+(?:success|experience|support|service|retention|satisfaction|health|lifecycle|journey|engagement|advocacy|feedback|voice|relationship|management|onboarding|training|education)\b',
            
            # Product & Platform
            r'\b(?:product|platform|solution|feature)\s+(?:adoption|onboarding|training|engagement|education|implementation|success|expertise|knowledge|roadmap|strategy|enhancement|development|management)\b',
            
            # Management & Leadership
            r'\b(?:stakeholder|relationship|account|portfolio|project|program|team|cross-functional|strategic|executive)\s+(?:management|leadership|coordination|alignment|communication|engagement)\b',
            
            # Metrics & Analytics
            r'\b(?:churn|revenue|renewal|retention|satisfaction|nps|csat|customer|client|account|arr|mrr|ltv|cac|qbr|roi)\s+(?:reduction|analysis|management|rate|score|metric|analytics|insights|reporting|tracking|monitoring|optimization|improvement|growth|performance|measurement|benchmark|target|goal|forecast|trend)s?\b',
            
            # Tools & Technology
            r'\b(?:salesforce|gainsight|zendesk|hubspot|intercom|freshdesk|totango|churnzero|vitally|planhat|jira|confluence|slack|zoom|teams|tableau|looker|powerbi|excel|google\s+analytics)\s+(?:admin|administration|configuration|customization|integration|development|automation|reporting|analytics|implementation|management|expertise|proficiency|experience|knowledge|skill)s?\b',
            
            # Process & Operations
            r'\b(?:business|process|workflow|service|operational|customer|client|onboarding|implementation|adoption|renewal|expansion|escalation|training)\s+(?:optimization|improvement|analysis|automation|transformation|efficiency|excellence|framework|methodology|strategy|management|coordination|execution|planning|development)s?\b',
            
            # Strategy & Growth
            r'\b(?:customer|retention|growth|success|revenue|business|market|product|account|portfolio|segment|territory|vertical|industry)\s+(?:strategy|planning|initiative|program|development|expansion|optimization|acceleration|management|analysis|segmentation|targeting|penetration|growth)s?\b',
            
            # Reviews & Performance
            r'\b(?:quarterly|monthly|weekly|annual|business|performance|success|executive|stakeholder|team)\s+(?:review|meeting|analysis|metric|scorecard|dashboard|reporting|presentation|communication|update|summary|report|assessment|evaluation)s?\b',
            
            # Health & Success Metrics
            r'\b(?:customer|client|account|product|user|platform|solution|service)\s+(?:health|success|satisfaction|adoption|usage|engagement|experience|journey|lifecycle|relationship)\s+(?:score|index|metric|indicator|analysis|tracking|monitoring|management|optimization|improvement)s?\b',
            
            # Specific CS/CX Metrics
            r'\b(?:net\s+promoter|customer\s+satisfaction|customer\s+effort|customer\s+health|time\s+to\s+value|first\s+response\s+time|resolution\s+time|customer\s+effort\s+score)\s+(?:score|index|rating|analysis|tracking|measurement|benchmark|improvement|optimization)s?\b',
            r'\b(?:monthly|annual)\s+(?:recurring\s+revenue|contract\s+value)\b',
            
            # Customer Journey
            r'\b(?:customer|client|user)\s+(?:journey|lifecycle|experience|success)\s+(?:mapping|optimization|analysis|management)\b',
            
            # Business Impact
            r'\b(?:revenue|business|account|customer)\s+(?:growth|expansion|optimization|impact|value|retention)\s+(?:strategy|program|initiative|analysis)s?\b',
            
            # Technical Skills
            r'\b(?:data|analytics|reporting|dashboard|integration|automation|api)\s+(?:development|implementation|management|analysis|configuration|customization)s?\b',
            
            # Soft Skills
            r'\b(?:stakeholder|executive|client|team|cross-functional)\s+(?:presentation|communication|management|collaboration|engagement|alignment)s?\b'
        ]
        
        # Extract multi-word skills using patterns
        for pattern in patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                skills.add(match.group())
        
        # Transform text for TF-IDF based extraction
        vector = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        scores = vector.toarray()[0]
        
        # Get non-zero scores
        non_zero_mask = scores > 0
        if any(non_zero_mask):
            non_zero_scores = scores[non_zero_mask]
            non_zero_indices = np.where(non_zero_mask)[0]
            if len(non_zero_scores) > 0:
                # Use percentile-based thresholding
                base_threshold = np.percentile(non_zero_scores, 60)  # Keep top 40%
                
                # Calculate final scores with context
                final_scores = defaultdict(float)
                for idx, base_score in zip(non_zero_indices, non_zero_scores):
                    term = feature_names[idx]
                    
                    # Find sentence containing the term
                    term_sentence = next((s for s in sentences if term.lower() in s.lower()), '')
                    context_score = sentence_scores.get(term_sentence, 1.0)
                    
                    # Apply domain-specific boosts
                    domain_boost = self._calculate_domain_boost(term)
                    
                    # Calculate final score
                    final_score = base_score * context_score * domain_boost
                    final_scores[term] = final_score
                
                # Process each non-zero score
                for idx, score in zip(non_zero_indices, non_zero_scores):
                    skill = feature_names[idx]
                    words = skill.split()
                    
                    # Adjust threshold based on skill characteristics
                    threshold = base_threshold
                    
                    # Lower threshold for domain-specific terms
                    if skill in self.synonyms or any(skill in syns for syns in self.synonyms.values()):
                        threshold *= 0.7
                    elif any(term in skill for term in self.domain_weights):
                        threshold *= 0.8
                    
                    # Lower threshold for CS/CX specific phrases
                    cs_phrases = {
                        'success': 0.7, 'experience': 0.7, 'management': 0.75,
                        'support': 0.75, 'satisfaction': 0.7, 'retention': 0.7,
                        'churn': 0.7, 'engagement': 0.75, 'implementation': 0.75,
                        'onboarding': 0.7, 'adoption': 0.7, 'revenue': 0.7,
                        'customer': 0.65, 'client': 0.65, 'account': 0.7,
                        'strategy': 0.75, 'analytics': 0.75, 'reporting': 0.75
                    }
                    
                    for phrase, threshold_mult in cs_phrases.items():
                        if phrase in skill:
                            threshold *= threshold_mult
                            break
                    
                    # Add skill if it meets criteria
                    if score >= threshold:
                        if len(words) <= 4:  # Allow up to 4-word phrases
                            skills.add(skill)
        
        # Extract domain terms
        domain_terms = {
            # Core Platforms & Tools
            'salesforce', 'gainsight', 'zendesk', 'intercom', 'hubspot', 'freshdesk', 'totango',
            'churnzero', 'vitally', 'planhat', 'catalyst', 'clientsuccess', 'custify', 'chargebee',
            'stripe', 'looker', 'tableau', 'powerbi', 'amplitude', 'mixpanel', 'pendo', 'fullstory',
            'jira', 'confluence', 'asana', 'notion', 'airtable', 'outreach', 'gong', 'chorus',
            
            # Core Metrics & KPIs
            'nps', 'csat', 'ces', 'churn', 'retention', 'mrr', 'arr', 'expansion', 'upsell', 'downsell',
            'qoq', 'yoy', 'roi', 'ltv', 'cac', 'arpu', 'logo', 'revenue', 'qlv', 'qrr', 'acv', 'tcv',
            'adoption', 'usage', 'engagement', 'health', 'satisfaction', 'effort', 'promoter',
            
            # Core Processes & Methodologies
            'onboarding', 'implementation', 'adoption', 'engagement', 'renewal', 'qbr', 'escalation',
            'enablement', 'deployment', 'rollout', 'migration', 'integration', 'playbook', 'framework',
            'lifecycle', 'journey', 'touchpoint', 'milestone', 'handoff', 'handover', 'transition',
            
            # Business & Strategy
        }
        
    def _calculate_context_score(self, sentence: str) -> float:
        """Calculate contextual relevance score for a sentence"""
        context_score = 1.0
        
        # Boost for achievement indicators
        achievement_indicators = [
            'improved', 'increased', 'reduced', 'managed', 'resolved',
            'optimized', 'enhanced', 'streamlined', 'implemented',
            'launched', 'developed', 'led', 'drove', 'achieved'
        ]
        if any(indicator in sentence.lower() for indicator in achievement_indicators):
            context_score *= 1.2
        
        # Boost for quantitative achievements
        if re.search(r'\d+%|\d+x|\$\d+|\d+k|\d+m', sentence.lower()):
            context_score *= 1.25
        
        # Boost for professional context
        professional_indicators = [
            'customer', 'client', 'account', 'team', 'project',
            'platform', 'solution', 'revenue', 'business', 'strategy'
        ]
        if any(indicator in sentence.lower() for indicator in professional_indicators):
            context_score *= 1.15
        
        return min(context_score, 2.0)  # Cap the boost
    
    def _validate_skill_coherence(self, skill: str) -> bool:
        """Validate if a skill is coherent in CS/CX context"""
        skill = skill.lower()
        words = skill.split()
        
        # Single word skills must be in domain vocabulary
        if len(words) == 1:
            # Check exact matches first
            if (skill in self.domain_weights or 
                skill in self.synonyms or 
                any(skill in syns for syns in self.synonyms.values())):
                return True
                
            # Check for partial matches in domain terms
            return any(domain_term in skill or skill in domain_term 
                      for domain_term in self.domain_weights.keys())
        
        # Check if multi-word skill follows valid CS/CX patterns
        valid_patterns = [
            # Core role patterns
            ['customer|client', 'success|experience|support|service|satisfaction|relationship|engagement'],
            ['account|portfolio', 'management|expansion|retention|health|strategy|growth'],
            ['product', 'adoption|implementation|training|enablement|support'],
            ['business|revenue', 'development|strategy|analysis|growth|optimization'],
            ['stakeholder|relationship', 'management|engagement|communication|development'],
            
            # Technical patterns
            ['technical|solution', 'implementation|integration|support|architecture'],
            ['data|analytics', 'analysis|reporting|visualization|insights'],
            
            # Process patterns
            ['project|program', 'management|coordination|delivery|execution'],
            ['change|process', 'management|improvement|optimization|automation'],
            
            # Leadership patterns
            ['team|department', 'leadership|management|development|coordination'],
            ['strategic|operational', 'planning|execution|management|development']
        ]
        
        # Check against patterns
        for pattern in valid_patterns:
            if re.search(f"\\b({pattern[0]})\\s+({pattern[1]})\\b", skill):
                return True
                
        # Check for domain-specific compound terms
        compound_terms = {
            'customer success', 'client success', 'account management',
            'customer experience', 'product adoption', 'revenue growth',
            'business development', 'stakeholder management', 'technical support',
            'data analysis', 'process improvement', 'team leadership'
        }
        
        return skill in compound_terms
        # Extract domain terms
        words = set(word_tokenize(text.lower()))
        skills.update(words & domain_terms)
        
        # Extract terms from domain weights
        for term in self.domain_weights:
            if re.search(r'\b' + re.escape(term) + r'\b', text.lower()):
                skills.add(term)
        
        # Normalize all extracted skills
        normalized_skills = {self._normalize_skill(skill) for skill in skills if skill}
        
        # Limit to top N skills
        if normalized_skills and len(normalized_skills) > self.max_skills:
            normalized_skills = set(sorted(normalized_skills, key=lambda s: final_scores.get(s, 0), reverse=True)[:self.max_skills])
        
        # Always return a set, even if empty
        return normalized_skills
    
    def get_skill_weight(self, skill: str) -> float:
        """Get weight based on TF-IDF importance"""
        # Convert skill to vector
        vector = self._get_text_vector(skill)
        
        # Use the maximum TF-IDF score as weight
        weight = float(vector.max())
        
        # Scale weight to be between 0.5 and 3.0
        scaled_weight = 0.5 + (weight * 2.5)
        
        return min(3.0, max(0.5, scaled_weight))
