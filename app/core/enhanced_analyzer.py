import logging
import re
import psutil
import time
import spacy
from typing import Any, List, Dict, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
import nltk
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
from datetime import datetime

# Import role-specific utilities
from app.core.role_patterns import (
    get_role_patterns,
    get_role_specific_skills,
    detect_role_type,
    calculate_role_specific_score
)

# Import monitoring utilities
from app.core.resource_monitor import ResourceMonitor
from app.core.resume_manager import ResumeManager
from collections import defaultdict
from .data_manager import DataManager, JobRequirements
from app.core.resource_monitor import ResourceMonitor
from app.utils.text_processor import TextProcessor

from functools import lru_cache

from app.utils.nltk_utils import ensure_nltk_data

# Ensure NLTK data is available
ensure_nltk_data()

@dataclass
class AnalysisResult:
    ats_score: float
    industry_match_score: float
    experience_match_score: float
    skill_matches: Dict[str, float]
    missing_critical_skills: List[str]
    improvement_suggestions: List[str]
    salary_match: bool
    location_match: bool
    overall_score: float
    optimized_resume: str = ""    # Optimized version of the resume
    keyword_density: Dict[str, float] = field(default_factory=dict)  # Keyword density analysis
    error: Optional[str] = None  # Track any errors during analysis
    processing_time: float = 0.0  # Track processing time
    memory_usage: float = 0.0    # Track peak memory usage
    role_type: Optional[str] = None  # Track role type (cx/cs)
    role_specific_metrics: Optional[Dict[str, int]] = field(default_factory=dict)  # Role-specific metrics

@dataclass
class TextPreprocessor:
    """Advanced text preprocessing for resume analysis."""
    lemmatizer: nltk.stem.WordNetLemmatizer
    stop_words: Set[str]
    
    @lru_cache(maxsize=1000)
    def preprocess(self, text: str) -> Dict[str, any]:
        """Preprocess text with advanced NLP techniques."""
        # Basic cleaning
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # spaCy processing
        doc = self.nlp(text)
        
        # Extract entities and noun phrases
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Tokenization with lemmatization
        tokens = [
            self.lemmatizer.lemmatize(token.text)
            for token in doc
            if not token.is_stop and not token.is_punct
        ]
        
        # Extract technical terms
        technical_terms = [
            token.text for token in doc
            if token.pos_ in {'NOUN', 'PROPN'} and
            not token.is_stop and len(token.text) > 2
        ]
        
        return {
            'clean_text': ' '.join(tokens),
            'entities': entities,
            'noun_phrases': noun_phrases,
            'technical_terms': technical_terms,
            'doc': doc
        }

class SemanticSearch:
    """Semantic search using transformer embeddings."""
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cache = {}
    
    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings."""
        # Tokenize and encode
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get embeddings
        outputs = self.model(**encoded)
        
        # Use mean pooling
        attention_mask = encoded['attention_mask']
        embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """Calculate mean pooling with attention mask."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def search(self, query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for most similar candidates to query."""
        if not candidates:
            return []
            
        # Encode query
        query_embedding = self.encode([query])
        
        # Encode candidates (with caching)
        candidate_embeddings = []
        new_candidates = []
        cached_indices = []
        final_candidates = []
        
        for i, candidate in enumerate(candidates):
            if candidate in self.cache:
                candidate_embeddings.append(self.cache[candidate])
                cached_indices.append(i)
                final_candidates.append(candidate)
            else:
                new_candidates.append(candidate)
        
        if new_candidates:
            try:
                new_embeddings = self.encode(new_candidates)
                for i, (candidate, embedding) in enumerate(zip(new_candidates, new_embeddings)):
                    self.cache[candidate] = embedding
                    candidate_embeddings.append(embedding)
                    final_candidates.append(candidate)
            except Exception as e:
                logging.error(f"Error encoding new candidates: {str(e)}")
                # Fall back to cached results only
                pass
        
        if not candidate_embeddings:
            return []
            
        try:
            # Stack embeddings
            candidate_embeddings = torch.stack(candidate_embeddings)
            
            # Calculate similarities
            similarities = F.cosine_similarity(query_embedding, candidate_embeddings)
            
            # Get top k results
            k = min(top_k, len(final_candidates))
            if k == 0:
                return []
                
            # Handle single result case
            if len(similarities.shape) == 0:
                return [(final_candidates[0], float(similarities))]
                
            top_k_scores, top_k_indices = similarities.topk(k)
            
            return [
                (final_candidates[idx], float(score))
                for idx, score in zip(top_k_indices.tolist(), top_k_scores.tolist())
            ]
        except Exception as e:
            logging.error(f"Error in similarity calculation: {str(e)}")
            return []

class EnhancedAnalyzer:
    # Role-specific skill patterns
    CX_CS_SKILL_PATTERNS = {
        'cx_ai_tools': [
            'chatbots', 'ai chatbots', 'conversational ai', 'machine learning models',
            'natural language processing', 'sentiment analysis', 'automated responses',
            'predictive analytics', 'ai-powered support', 'intelligent automation'
        ],
        'cx_metrics': [
            'csat', 'customer satisfaction', 'nps', 'net promoter score',
            'ai accuracy rates', 'automation success rate', 'response accuracy',
            'ml model performance', 'automation efficiency', 'prediction accuracy'
        ],
        'cx_processes': [
            'ai model training', 'data annotation', 'model deployment',
            'automated ticket routing', 'intelligent escalation', 'automated classification',
            'ml pipeline management', 'model monitoring', 'ai ethics compliance'
        ],
        'cs_ai_tools': [
            'predictive churn models', 'customer health ai', 'revenue prediction',
            'usage pattern analysis', 'anomaly detection', 'recommendation systems',
            'automated onboarding', 'engagement scoring', 'propensity modeling'
        ],
        'cs_metrics': [
            'model accuracy', 'prediction success rate', 'automation roi',
            'ai adoption rate', 'customer health predictions', 'churn predictions',
            'revenue forecasting accuracy', 'engagement scores', 'automation coverage'
        ],
        'cs_processes': [
            'ai strategy development', 'model lifecycle management',
            'automated customer journey', 'predictive success planning',
            'ai-driven risk management', 'automated expansion detection',
            'intelligent stakeholder updates', 'automated business reviews'
        ]
    }
    
    def __init__(self, data_manager: DataManager):
        """Initialize with enhanced NLP capabilities."""
        # Initialize base components
        self.logger = logging.getLogger(__name__)
        self.data_manager = data_manager
        self.monitor = ResourceMonitor()
        
        # Initialize NLP components
        self.text_processor = TextProcessor()
        
        # Add preprocess method for backward compatibility
        def preprocess(self_, text):
            return self.text_processor.preprocess_text(text)
        self.text_preprocessor = type('TextPreprocessor', (), {'preprocess': preprocess})()
        
        # Initialize role patterns
        self.role_patterns = {
            'cx': {
                'keywords': [
                    'customer experience', 'cx', 'customer satisfaction', 'customer journey',
                    'voice of customer', 'voc', 'user experience', 'ux', 'customer feedback',
                    'customer insight', 'customer analytics', 'customer research', 'usability',
                    'customer sentiment', 'customer behavior', 'customer engagement'
                ],
                'skills': [
                    'empathy', 'communication', 'problem-solving', 'analytics',
                    'journey mapping', 'service design', 'data visualization',
                    'stakeholder management', 'process improvement', 'user research',
                    'experience design', 'customer advocacy', 'project management',
                    'workshop facilitation', 'strategy development', 'cross-functional leadership'
                ],
                'tools': [
                    'zendesk', 'intercom', 'salesforce', 'qualtrics', 'medallia',
                    'surveymonkey', 'usertesting', 'hotjar', 'fullstory', 'optimizely',
                    'google analytics', 'tableau', 'power bi', 'miro', 'figma',
                    'jira', 'confluence', 'asana', 'clickup', 'notion'
                ],
                'metrics': [
                    'nps', 'csat', 'ces', 'customer effort score', 'time to resolution',
                    'first contact resolution', 'customer satisfaction score',
                    'customer retention rate', 'customer churn rate', 'engagement score',
                    'conversion rate', 'bounce rate', 'session duration'
                ]
            },
            'cs': {
                'keywords': [
                    'customer success', 'cs', 'account management', 'client retention',
                    'customer relationship', 'client success', 'customer support',
                    'technical support', 'client management', 'customer service',
                    'account growth', 'revenue retention', 'customer lifecycle'
                ],
                'skills': [
                    'relationship management', 'onboarding', 'product expertise',
                    'business acumen', 'account planning', 'risk management',
                    'revenue growth', 'customer training', 'technical troubleshooting',
                    'sales enablement', 'contract negotiation', 'project management',
                    'data analysis', 'strategic planning', 'executive presentation'
                ],
                'tools': [
                    'gainsight', 'totango', 'churnzero', 'clientsuccess', 'salesforce',
                    'hubspot', 'zendesk', 'jira', 'confluence', 'slack', 'zoom',
                    'microsoft teams', 'looker', 'tableau', 'power bi', 'asana',
                    'monday.com', 'freshdesk', 'intercom', 'drift'
                ],
                'metrics': [
                    'mrr', 'arr', 'net revenue retention', 'gross revenue retention',
                    'logo retention', 'churn rate', 'expansion revenue', 'time to value',
                    'product adoption rate', 'customer health score', 'qbr effectiveness',
                    'renewal rate', 'upsell rate', 'customer lifetime value'
                ]
            }
        }
        
        # Initialize device
        self.logger.debug("Initializing device")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize ML models
        self.logger.debug("Initializing ML models")
        try:
            # Initialize BERT for embeddings
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
            
            # Initialize sentiment analyzer
            self.logger.debug("Initializing sentiment analyzer")
            self.sentiment_analyzer = pipeline(
                'sentiment-analysis',
                model='distilbert-base-uncased-finetuned-sst-2-english',
                device=self.device
            )
            
            # Initialize weights for scoring
            self.weights = {
                'skills': 0.35,
                'experience': 0.25,
                'title': 0.2,
                'industry': 0.1,
                'location': 0.05,
                'salary': 0.05
            }
            
            # Initialize semantic similarity threshold
            self.similarity_threshold = 0.85
            
            # Initialize zero-shot classifier
            self.logger.debug("Initializing zero-shot classifier")
            self.zero_shot = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device
            )
            
            # Initialize semantic search
            self.logger.debug("Initializing semantic search")
            self.semantic_search = SemanticSearch(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            
            # Initialize spaCy
            self.logger.debug("Initializing spaCy NLP")
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                self.logger.info("Downloading spaCy model...")
                spacy.cli.download('en_core_web_sm')
                self.nlp = spacy.load('en_core_web_sm')
            
            self.logger.info("ML models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing ML models: {str(e)}")
            raise RuntimeError(f"Failed to initialize ML models: {str(e)}")
        
    def _init_base_components(self, data_manager: DataManager):
        """
        Initialize the EnhancedAnalyzer with optimized memory management.
        
        This constructor handles several critical initialization steps:
        1. Sets up logging and monitoring
        2. Initializes memory thresholds and tracking
        3. Sets up NLP components
        4. Configures the ML model with memory optimization
        
        Args:
            data_manager: DataManager instance for handling job data
            
        Raises:
            RuntimeError: If critical initialization steps fail
        """
        self.logger = logging.getLogger(__name__)
        self.data_manager = data_manager
        self.monitor = ResourceMonitor()
        
        # Role-specific patterns with comprehensive coverage
        self.role_patterns = {
            'cx': {
                'keywords': [
                    'customer experience', 'cx', 'customer satisfaction', 'customer journey',
                    'voice of customer', 'voc', 'user experience', 'ux', 'customer feedback',
                    'customer insight', 'customer analytics', 'customer research', 'usability',
                    'customer sentiment', 'customer behavior', 'customer engagement'
                ],
                'skills': [
                    'empathy', 'communication', 'problem-solving', 'analytics',
                    'journey mapping', 'service design', 'data visualization',
                    'stakeholder management', 'process improvement', 'user research',
                    'experience design', 'customer advocacy', 'project management',
                    'workshop facilitation', 'strategy development', 'cross-functional leadership'
                ],
                'tools': [
                    'zendesk', 'intercom', 'salesforce', 'qualtrics', 'medallia',
                    'surveymonkey', 'usertesting', 'hotjar', 'fullstory', 'optimizely',
                    'google analytics', 'tableau', 'power bi', 'miro', 'figma',
                    'jira', 'confluence', 'asana', 'clickup', 'notion'
                ],
                'metrics': [
                    'nps', 'csat', 'ces', 'customer effort score', 'time to resolution',
                    'first contact resolution', 'customer satisfaction score',
                    'customer retention rate', 'customer churn rate', 'engagement score',
                    'conversion rate', 'bounce rate', 'session duration'
                ]
            },
            'cs': {
                'keywords': [
                    'customer success', 'cs', 'account management', 'client retention',
                    'customer relationship', 'client success', 'customer support',
                    'technical support', 'client management', 'customer service',
                    'account growth', 'revenue retention', 'customer lifecycle'
                ],
                'skills': [
                    'relationship management', 'onboarding', 'product expertise',
                    'business acumen', 'account planning', 'risk management',
                    'revenue growth', 'customer training', 'technical troubleshooting',
                    'sales enablement', 'contract negotiation', 'project management',
                    'data analysis', 'strategic planning', 'executive presentation'
                ],
                'tools': [
                    'gainsight', 'totango', 'churnzero', 'clientsuccess', 'salesforce',
                    'hubspot', 'zendesk', 'jira', 'confluence', 'slack', 'zoom',
                    'microsoft teams', 'looker', 'tableau', 'power bi', 'asana',
                    'monday.com', 'freshdesk', 'intercom', 'drift'
                ],
                'metrics': [
                    'mrr', 'arr', 'net revenue retention', 'gross revenue retention',
                    'logo retention', 'churn rate', 'expansion revenue', 'time to value',
                    'product adoption rate', 'customer health score', 'qbr effectiveness',
                    'renewal rate', 'upsell rate', 'customer lifetime value'
                ]
            }
        }
        
        # Batch processing configuration
        self.batch_config = {
            'max_batch_size': 4,  # Maximum number of resumes per batch
            'chunk_size': 512,    # Maximum tokens per text chunk
            'memory_threshold': 0.85,  # Memory threshold for batch processing
            'max_retries': 3      # Maximum retries for failed batches
        }

        """Initialize the EnhancedAnalyzer with optimized memory management.
        
        This constructor handles several critical initialization steps:
        1. Sets up logging and monitoring
        2. Initializes memory thresholds and tracking
        3. Sets up NLP components
        4. Configures the ML model with memory optimization
        
        Args:
            data_manager: DataManager instance for handling job data
            
        Raises:
            RuntimeError: If critical initialization steps fail
        """
        self.logger = logging.getLogger(__name__)
        self.data_manager = data_manager
        self.monitor = ResourceMonitor()
        
        # Initialize memory tracking with detailed thresholds
        try:
            vm = psutil.virtual_memory()
            total_memory = vm.total
            available_memory = vm.available
            
            # Calculate dynamic thresholds based on system state
            base_threshold = 0.7  # 70% base threshold
            critical_threshold = 0.85  # 85% critical threshold
            
            # Adjust thresholds if system is already under memory pressure
            current_usage = (total_memory - available_memory) / total_memory
            if current_usage > base_threshold:
                self.logger.warning(
                    f"System already under memory pressure: {current_usage:.1f}% used"
                )
                base_threshold = current_usage + 0.1  # Add 10% margin
                critical_threshold = current_usage + 0.15  # Add 15% margin
            
            self.memory_threshold = base_threshold * total_memory
            self.memory_critical = critical_threshold * total_memory
            self.initial_memory = psutil.Process().memory_info().rss
            
            self.logger.info(
                f"Memory thresholds set - Warning: {base_threshold*100:.1f}%, "
                f"Critical: {critical_threshold*100:.1f}%"
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing memory tracking: {str(e)}")
            raise RuntimeError(f"Failed to initialize memory tracking: {str(e)}")
        
        # Initialize NLTK components with error recovery
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except LookupError as e:
            self.logger.warning(f"NLTK data missing: {str(e)}. Attempting download.")
            try:
                nltk.download('punkt')
                nltk.download('stopwords')
                nltk.download('wordnet')
                nltk.download('averaged_perceptron_tagger')
                
                # Retry initialization
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            except Exception as e:
                self.logger.error(f"Failed to download NLTK data: {str(e)}")
                raise RuntimeError(f"Failed to initialize NLTK components: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error initializing NLTK components: {str(e)}")
            raise RuntimeError(f"Failed to initialize NLTK: {str(e)}")
        
        # Initialize ML models with advanced configurations
        try:
            # Use a more sophisticated model for technical content
            model_name = 'microsoft/mpnet-base'  # Better semantic understanding
            self.logger.info(f"Loading primary model {model_name}")
            
            # Initialize device with optimal settings
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
                torch.mps.set_per_process_memory_fraction(0.7)  # Prevent memory issues
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
                torch.cuda.set_per_process_memory_fraction(0.7)
            else:
                self.device = torch.device('cpu')
            self.logger.info(f"Primary device: {self.device}")
            
            # Enhanced model configuration
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.model_max_length = 512
            
            # Initialize primary model with optimizations
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                return_dict=True
            )
            
            # Try to move to better device after initialization
            try:
                better_device = self._initialize_device()
                if better_device.type != 'cpu':
                    self.model = self.model.to(better_device)
                    self.device = better_device
                    self.logger.info(f"Successfully moved model to {self.device}")
            except Exception as e:
                self.logger.warning(f"Could not move model to better device: {str(e)}. Staying on CPU.")
            
            # Core performance settings
            self.model.eval()  # Ensure evaluation mode
            torch.set_grad_enabled(False)  # Disable gradients globally
            
        except Exception as e:
            self.logger.error(f"Error initializing transformer model: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
        
        # Initialize TF-IDF vectorizer for technical content
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),  # Capture multi-word technical terms
            max_features=10000,   # Increased vocabulary
            min_df=2,            # Minimum document frequency
            max_df=0.95,         # Maximum document frequency
            token_pattern=r'(?u)\b\w[\w\.+#-]*\w\b',  # Better for technical terms
            dtype=np.float32     # Use float32 for memory efficiency
        )
        
        # Score weights
        self.weights = {
            'skills': 0.4,
            'experience': 0.3,
            'industry': 0.2,
            'location': 0.05,
            'salary': 0.05
        }

    def _initialize_device(self) -> torch.device:
        """Initialize device with consistent properties."""
        if torch.backends.mps.is_available():
            # Explicitly create device without index for consistency
            return torch.device('mps')
        return torch.device('cpu')
    
    def _verify_device_compatibility(self, tensor: torch.Tensor) -> bool:
        """Verify tensor device compatibility without strict equality."""
        return tensor.device.type == self.device.type
    
    def optimize_resume(self, resume_text: str, job_description: str) -> str:
        """Optimize the resume text based on the job description.
        
        Args:
            resume_text: The original resume text
            job_description: The job description to optimize against
            
        Returns:
            str: The optimized resume text
        """
        try:
            # Analyze the resume first
            analysis = self.analyze_resume(resume_text, job_description)
            
            # Get the original resume lines
            resume_lines = resume_text.split('\n')
            optimized_lines = resume_lines.copy()
            
            # Track changes for each section
            changes_made = []
            
            # Optimize skills section
            if analysis.missing_critical_skills:
                # Find the skills section
                skills_idx = -1
                for i, line in enumerate(resume_lines):
                    if re.search(r'skills|technical skills|core competencies', line.lower()):
                        skills_idx = i
                        break
                
                if skills_idx >= 0:
                    # Add missing skills with a note
                    missing_skills = ", ".join(analysis.missing_critical_skills)
                    optimized_lines.insert(skills_idx + 1, 
                        f"Actively developing: {missing_skills}")
                    changes_made.append("Added missing critical skills section")
            
            # Optimize based on keyword density
            if hasattr(analysis, 'keyword_density'):
                for keyword, density in analysis.keyword_density.items():
                    if density < 0.01:  # Less than 1% density
                        # Look for opportunities to add the keyword
                        for i, line in enumerate(optimized_lines):
                            if keyword.lower() in line.lower():
                                # Enhance the line with more context
                                context = self._get_keyword_context(keyword, job_description)
                                if context:
                                    optimized_lines[i] = f"{line} ({context})"
                                    changes_made.append(f"Enhanced context for '{keyword}'")
                                break
            
            # Add a summary of changes at the top
            if changes_made:
                optimized_lines.insert(0, "")
                optimized_lines.insert(0, "Optimization Notes:")
                for change in changes_made:
                    optimized_lines.insert(1, f"• {change}")
                optimized_lines.insert(len(changes_made) + 2, "")
            
            return '\n'.join(optimized_lines)
            
        except Exception as e:
            self.logger.error(f"Error optimizing resume: {str(e)}", exc_info=True)
            return resume_text  # Return original if optimization fails
    
    def _get_keyword_context(self, keyword: str, job_description: str) -> str:
        """Get relevant context for a keyword from the job description."""
        try:
            # Find sentences containing the keyword
            sentences = sent_tokenize(job_description)
            relevant_sentences = [s for s in sentences if keyword.lower() in s.lower()]
            
            if relevant_sentences:
                # Get the shortest relevant sentence for concise context
                context = min(relevant_sentences, key=len)
                # Clean and shorten the context
                context = re.sub(r'[\[\](){}<>]', '', context)
                context = context.strip()
                if len(context) > 100:
                    context = context[:97] + "..."
                return context
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Error getting keyword context: {str(e)}")
            return ""
    def analyze_batch(self,
                    resumes: List[str],
                    job_description: str,
                    target_job: Optional[str] = None,
                    target_industry: Optional[str] = None) -> List[AnalysisResult]:
        """
        Analyze multiple resumes against a job description with efficient batching.
        
        This method implements memory-efficient batch processing:
        1. Dynamic batch sizing based on system resources
        2. Parallel processing where available
        3. Automatic error recovery
        4. Progress tracking and logging
        
        Args:
            resumes: List of resume texts to analyze
            job_description: Job description to match against
            target_job: Optional specific job title
            target_industry: Optional specific industry
            
        Returns:
            List[AnalysisResult]: Analysis results for each resume
        """
        results = []
        total_resumes = len(resumes)
        batch_size = self.batch_config['max_batch_size']
        
        self.logger.info(f"Starting batch analysis of {total_resumes} resumes")
        start_time = time.time()
        
        try:
            # Find matching requirements once for all resumes
            job_req = self._find_matching_requirements(
                job_description, target_job, target_industry
            )
            
            if not job_req:
                raise ValueError("Could not match job description to known patterns")
            
            # Process resumes in batches
            for batch_start in range(0, total_resumes, batch_size):
                batch_end = min(batch_start + batch_size, total_resumes)
                batch = resumes[batch_start:batch_end]
                
                self.logger.info(
                    f"Processing batch {batch_start//batch_size + 1}/"
                    f"{(total_resumes + batch_size - 1)//batch_size} "
                    f"(resumes {batch_start + 1}-{batch_end}/{total_resumes})"
                )
                
                # Check memory before processing batch
                if self._monitor_memory_usage() > self.batch_config['memory_threshold']:
                    self._clear_cache(force=True)
                
                # Process each resume in the batch
                batch_results = []
                for resume_text in batch:
                    try:
                        result = self.analyze_resume(
                            resume_text, job_description,
                            target_job, target_industry
                        )
                        batch_results.append(result)
                    except Exception as e:
                        error_msg = f"Error processing resume: {str(e)}"
                        self.logger.error(error_msg)
                        # Create error result
                        error_result = AnalysisResult(
                            ats_score=0.0,
                            industry_match_score=0.0,
                            experience_match_score=0.0,
                            skill_matches={},
                            missing_critical_skills=[],
                            improvement_suggestions=["Analysis failed"],
                            salary_match=False,
                            location_match=False,
                            overall_score=0.0,
                            error=error_msg
                        )
                        batch_results.append(error_result)
                
                results.extend(batch_results)
                
                # Log batch completion
                elapsed = time.time() - start_time
                avg_time = elapsed / (batch_end)
                remaining = (total_resumes - batch_end) * avg_time
                
                self.logger.info(
                    f"Batch complete. Avg time per resume: {avg_time:.1f}s. "
                    f"Estimated remaining time: {remaining:.1f}s"
                )
            
            return results
            
        except Exception as e:
            error_msg = f"Batch processing failed: {str(e)}"
            self.logger.error(error_msg)
            # Return error results for all remaining resumes
            while len(results) < total_resumes:
                error_result = AnalysisResult(
                    ats_score=0.0,
                    industry_match_score=0.0,
                    experience_match_score=0.0,
                    skill_matches={},
                    missing_critical_skills=[],
                    improvement_suggestions=["Batch processing failed"],
                    salary_match=False,
                    location_match=False,
                    overall_score=0.0,
                    error=error_msg
                )
                results.append(error_result)
            return results

    def _detect_role_type(self, job_description: str) -> str:
        """Detect if the job is for CX or CS role.
        
        Args:
            job_description: Job posting text
            
        Returns:
            str: 'cx' or 'cs' or None if not detected
        """
        try:
            # Convert to lowercase for matching
            text = job_description.lower()
            
            # Count CX/CS related terms
            cx_terms = ['customer experience', 'cx', 'user experience', 'ux']
            cs_terms = ['customer success', 'cs', 'customer support', 'account management']
            
            cx_count = sum(text.count(term) for term in cx_terms)
            cs_count = sum(text.count(term) for term in cs_terms)
            
            self.logger.debug(f"Role detection counts - CX: {cx_count}, CS: {cs_count}")
            
            # Determine role type based on term frequency
            if cx_count > cs_count:
                return 'cx'
            elif cs_count > cx_count:
                return 'cs'
            elif cx_count > 0:  # If tied but has CX terms
                return 'cx'
            elif cs_count > 0:  # If tied but has CS terms
                return 'cs'
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error in role detection: {str(e)}")
            return None
            
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from text using NLP techniques.
        
        Args:
            text: Input text to extract skills from
            
        Returns:
            List[str]: Extracted skills
        """
        try:
            # Process text with spaCy
            doc = self.nlp(text.lower())
            
            # Extract noun phrases as potential skills
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            
            # Get role-specific skills
            role_skills = set()
            for role in ['cx', 'cs']:
                if role in self.role_patterns:
                    role_skills.update(self.role_patterns[role]['skills'])
                    role_skills.update(self.role_patterns[role]['tools'])
            
            # Match skills
            matched_skills = set()
            for phrase in noun_phrases:
                if phrase in role_skills:
                    matched_skills.add(phrase)
                    
            self.logger.debug(f"Extracted {len(matched_skills)} skills")
            return list(matched_skills)
            
        except Exception as e:
            self.logger.error(f"Error in skill extraction: {str(e)}")
            return []
            
    def _calculate_skill_matches(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Calculate skill matches between resume and job description.
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            
        Returns:
            Dict with match scores and details
        """
        try:
            # Extract skills
            resume_skills = self._extract_skills(resume_text)
            job_skills = self._extract_skills(job_description)
            
            # Calculate matches
            matches = set(resume_skills) & set(job_skills)
            missing = set(job_skills) - set(resume_skills)
            
            # Calculate match score
            if job_skills:
                match_score = len(matches) / len(job_skills)
            else:
                match_score = 0.0
                
            result = {
                'matches': list(matches),
                'missing_skills': list(missing),
                'match_score': match_score
            }
            
            self.logger.debug("Skill match results: %s", result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error in skill matching: {str(e)}")
            return {'matches': [], 'missing_skills': [], 'match_score': 0.0}
            
    def _calculate_experience_match(self, resume_text: str, job_description: str) -> float:
        """Calculate experience match score.
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            
        Returns:
            float: Experience match score between 0 and 1
        """
        try:
            # Extract years of experience from both texts
            resume_years = self._extract_years_experience(resume_text)
            required_years = self._extract_years_experience(job_description)
            
            if required_years == 0:
                return 1.0  # No experience requirement specified
                
            if resume_years >= required_years:
                return 1.0
            elif resume_years >= required_years * 0.75:
                return 0.75
            elif resume_years >= required_years * 0.5:
                return 0.5
            else:
                return 0.25
                
        except Exception as e:
            self.logger.error(f"Error in experience matching: {str(e)}")
            return 0.0
            
    def _extract_years_experience(self, text: str) -> int:
        """Extract years of experience from text.
        
        Args:
            text: Input text
            
        Returns:
            int: Years of experience (0 if not found)
        """
        try:
            # Common patterns for years of experience
            patterns = [
                r'(\d+)\+?\s*(?:years?|yrs?).{0,20}(?:experience|work)',
                r'(?:experience|work).{0,20}(\d+)\+?\s*(?:years?|yrs?)',
            ]
            
            text = text.lower()
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    # Take the maximum years if multiple matches
                    return max(int(match) for match in matches)
                    
            return 0
            
        except Exception as e:
            self.logger.error(f"Error extracting years: {str(e)}")
            return 0
            
    def analyze_resume(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Analyze resume with enhanced NLP techniques and role-specific metrics.
        
        This method performs comprehensive resume analysis including:
        1. Skill matching using TF-IDF and semantic search
        2. Experience and industry matching
        3. Role-specific metrics for CX/CS positions
        4. Resume optimization suggestions
        
        Args:
            resume_text: The resume text to analyze
            job_description: Job posting to match against
            
        Returns:
            Dict containing analysis results including:
            - ats_score: Overall ATS score between 0 and 1
            - keyword_density: Density of relevant keywords
            - skill_matches: Dictionary of matched skills
            - experience_score: Experience match score
            - suggestions: List of improvement suggestions
            - role_type: Detected role type (cx/cs)
        """
        try:
            # Validate inputs
            if not isinstance(resume_text, str) or not resume_text or not resume_text.strip():
                raise ValueError("Resume text must be a non-empty string")
            if not isinstance(job_description, str) or not job_description or not job_description.strip():
                raise ValueError("Job description must be a non-empty string")
                
            # Log validated inputs
            self.logger.debug("Starting analyze_resume with:")
            self.logger.debug(f"Resume text length: {len(resume_text)}")
            self.logger.debug(f"Job description length: {len(job_description)}")
                
            # Detect role type
            role_type = self._detect_role_type(job_description)
            self.logger.info("Detected role type: %s", role_type)
                
            # Extract skills from both texts
            resume_skills = self._extract_skills(resume_text)
            job_skills = self._extract_skills(job_description)
            self.logger.debug("Extracted skills - Resume: %s, Job: %s", resume_skills, job_skills)
            
            # Calculate skill matches
            skill_matches = self._calculate_skill_matches(resume_text, job_description)
            self.logger.debug("Skill matches: %s", skill_matches)
            
            # Extract and calculate experience scores
            experience_score = self._calculate_experience_match(resume_text, job_description)
            self.logger.debug("Experience score: %f", experience_score)
            self.logger.debug("Experience score: %f", experience_score)
            
            # Calculate keyword density
            keyword_density = self._calculate_keyword_density(resume_text, job_skills)
            self.logger.debug("Keyword density: %f", keyword_density)
            
            # Calculate role-specific metrics if role type detected
            role_metrics = None
            if role_type:
                role_metrics = self._calculate_role_metrics(resume_text, job_description, role_type)
                self.logger.debug("Role-specific metrics: %s", role_metrics)
            
            # Calculate additional scores
            industry_score = self._calculate_industry_match(resume_text, job_description)
            location_match = self._check_location_match(resume_text, job_description)
            salary_match = self._check_salary_match(resume_text, job_description)
            title_match = self._calculate_title_match(resume_text, job_description)

            # Calculate final ATS score
            skill_matches_dict = skill_matches if isinstance(skill_matches, dict) else {'overall': skill_matches}
            ats_score = self._calculate_ats_score(
                skill_matches=skill_matches_dict,
                experience_score=experience_score,
                industry_score=industry_score,
                location_match=location_match,
                salary_match=salary_match,
                title_match=title_match,
                role_metrics=role_metrics
            )
            self.logger.debug("Final ATS score: %f", ats_score)
            
            # Get job requirements
            job_req = self._find_matching_requirements(job_description)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(
                skill_matches=skill_matches,
                experience_score=experience_score,
                industry_score=industry_score,
                location_match=location_match,
                salary_match=salary_match,
                job_req=job_req,
                role_metrics=role_metrics
            )
            
            # Prepare result dictionary
            result = {
                'ats_score': ats_score,
                'keyword_density': keyword_density,
                'skill_matches': skill_matches,
                'experience_score': experience_score,
                'suggestions': suggestions,
                'role_type': role_type,
                'role_metrics': role_metrics
            }
            
            self.logger.info("Resume analysis completed successfully")
            return result
            
        except Exception as e:
            self.logger.error("Error in analyze_resume: %s", str(e), exc_info=True)
            raise
            raise

                
            # Get job requirements
            job_req = self._find_matching_requirements(job_description)
            if not job_req:
                raise ValueError("Failed to extract job requirements")
            
            # Calculate original metrics
            skill_matches = self._calculate_skill_matches(self.master_resume, job_req)
            experience_score = self._calculate_experience_match(self.master_resume, job_req)
            industry_score = self._calculate_industry_match(self._get_text_embedding(self.master_resume), job_req)
            location_match = self._check_location_match(self.master_resume, job_req)
            salary_match = self._check_salary_match(self.master_resume, job_req)
            title_match = self._calculate_title_match(self.master_resume, job_req.title)
            
            # Store all metrics in a dictionary
            metrics = {
                'skill_matches': skill_matches,
                'experience_score': float(experience_score),
                'industry_score': float(industry_score),
                'location_match': bool(location_match),
                'salary_match': bool(salary_match),
                'title_match': float(title_match),
                'ats_score': float(original_ats)
            }
            self.logger.debug("Metrics dictionary: %s", metrics)
            
            # Debug logging for all components
            self.logger.debug("Skill matches: %s (type: %s)", skill_matches, type(skill_matches))
            self.logger.debug("Experience score: %s (type: %s)", experience_score, type(experience_score))
            self.logger.debug("Industry score: %s (type: %s)", industry_score, type(industry_score))
            self.logger.debug("Location match: %s (type: %s)", location_match, type(location_match))
            self.logger.debug("Salary match: %s (type: %s)", salary_match, type(salary_match))
            self.logger.debug("Title match: %s (type: %s)", title_match, type(title_match))
            
            # Calculate original ATS score
            original_ats = self._calculate_ats_score(
                skill_matches,
                experience_score,
                industry_score,
                location_match,
                salary_match,
                title_match
            )
            
            self.logger.debug("Original ATS score: %s (type: %s)", original_ats, type(original_ats))
            
            # Calculate original keyword density
            original_density = self._calculate_keyword_density(self.master_resume, list(job_req.required_skills))
            
            # Create result dictionary
            result = {
                'ats_score': original_ats * 100,  # Convert to percentage
                'skill_matches': skill_matches,
                'experience_score': experience_score,
                'industry_score': industry_score,
                'location_match': location_match,
                'salary_match': salary_match,
                'keyword_density': original_density,
                'suggestions': [],
                'missing_skills': []
            }
            
            # Optimize resume
            missing_skills = self._identify_missing_skills(skill_matches)
            result['missing_skills'] = missing_skills
            suggestions = self._generate_suggestions(
                skill_matches,
                experience_score,
                industry_score,
                location_match,
                salary_match,
                job_req
            )
            optimized = self._optimize_resume_text(self.master_resume, missing_skills, suggestions, skill_matches)
            
            # Calculate new metrics
            new_skill_matches = self._calculate_skill_matches(optimized, job_req)
            new_ats = self._calculate_ats_score(
                new_skill_matches,
                experience_score,
                industry_score,
                location_match,
                salary_match,
                title_match
            )
            new_density = self._calculate_keyword_density(optimized, list(job_req.required_skills))
            
            # Update result with optimized values
            result.update({
                'optimized_text': optimized,
                'ats_score': new_ats * 100,  # Convert to percentage
                'keyword_density': new_density
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in resume analysis: {str(e)}", exc_info=True)
            raise
        try:
            start_time = time.time()
            initial_memory = self._monitor_memory_usage()
            
            # Preprocess texts
            processed_resume = self.text_processor.preprocess_text(resume_text)
            processed_job = self.text_processor.preprocess_text(job_description)
            
            # Get job requirements
            job_req = self._find_matching_requirements(
                job_description,
                target_job,
                target_industry
            )
            
            if not job_req:
                return AnalysisResult(
                    ats_score=0.0,
                    industry_match_score=0.0,
                    experience_match_score=0.0,
                    skill_matches={},
                    missing_critical_skills=[],
                    improvement_suggestions=[],
                    salary_match=False,
                    location_match=False,
                    overall_score=0.0,
                    optimized_resume="",
                    keyword_density={},
                    error="Failed to extract job requirements",
                    processing_time=time.time() - start_time,
                    memory_usage=self._monitor_memory_usage() - initial_memory,
                    role_type=target_job,
                    role_specific_metrics={}
                )
            
            # Calculate title match
            title_match = self._calculate_title_match(resume_text, job_req.title)
            
            # Calculate skill matches with semantic search
            all_skills = job_req.required_skills.union(job_req.priority_skills)
            skill_scores = {}
            
            # Direct matches
            for skill in all_skills:
                if skill.lower() in processed_resume['clean_text']:
                    skill_scores[skill] = 1.0
                    
            # Semantic matches
            resume_embedding = self._get_text_embedding(processed_resume['clean_text'])
            for skill in all_skills:
                if skill not in skill_scores:  # Only check if not already matched
                    skill_embedding = self._get_text_embedding(skill)
                    if skill_embedding is not None and resume_embedding is not None:
                        similarity = F.cosine_similarity(resume_embedding, skill_embedding, dim=1)
                        if similarity > self.similarity_threshold:
                            skill_scores[skill] = float(similarity.item())
            
            # Add enterprise experience detection
            if 'enterprise' in job_req.title.lower():
                enterprise_terms = ['enterprise', 'large accounts', 'strategic accounts', 'key accounts']
                for term in enterprise_terms:
                    if term in processed_resume['clean_text']:
                        skill_scores['enterprise experience'] = 1.0
                        break
            
            # Calculate original metrics
            original_density = self._calculate_keyword_density(resume_text, list(job_req.required_skills))
            original_ats = self._calculate_ats_score(
                skill_matches=skill_scores,
                experience_score=self._calculate_experience_match(resume_text, job_req),
                industry_score=self._calculate_industry_match(resume_embedding, job_req),
                location_match=self._check_location_match(resume_text, job_req),
                salary_match=self._check_salary_match(resume_text, job_req),
                title_match=title_match
            )
            
            # Optimize resume to hit targets
            optimized_resume, new_metrics = self._optimize_resume(
                resume_text=resume_text,
                job_description=job_description,
                original_metrics={'ats': original_ats, 'density': original_density}
            )
            location_match = self._check_location_match(resume_text, job_req)
            salary_match = self._check_salary_match(resume_text, job_req)
            
            # Calculate role-specific metrics if job type detected
            role_type = None
            if re.search(r'(?i)(customer experience|cx|customer support|help desk|technical support)', job_description):
                role_type = 'cx'
            elif re.search(r'(?i)(customer success|cs manager|csm|client success)', job_description):
                role_type = 'cs'
            
            role_metrics = None
            if role_type:
                role_metrics = self._calculate_role_metrics(
                    resume_text,
                    job_description,
                    role_type
                )
            
            # Calculate ATS score with role metrics and title match
            ats_score = self._calculate_ats_score(
                skill_scores,
                experience_score,
                industry_score,
                location_match,
                salary_match,
                title_match,
                role_metrics
            )
            
            # Calculate keyword density
            keyword_density = self._calculate_keyword_density(
                processed_resume['clean_text'],
                processed_job['clean_text']
            )
            
            # Identify missing skills
            missing_skills = self._identify_missing_skills(skill_scores)
            
            # Generate suggestions with role metrics
            suggestions = self._generate_suggestions(
                skill_scores,
                experience_score,
                industry_score,
                location_match,
                salary_match,
                job_req,
                role_metrics
            )
            
            # Optimize resume
            optimized_resume = self._optimize_resume(
                resume_text,
                missing_skills,
                suggestions,
                skill_scores
            )
            
            # Calculate role-specific metrics
            role_metrics = self._calculate_role_metrics(
                processed_resume,
                processed_job,
                job_req.role_type
            )
            
            # Create result with all fields
            result = AnalysisResult(
                ats_score=ats_score,
                industry_match_score=industry_score,
                experience_match_score=experience_score,
                skill_matches=skill_scores,
                missing_critical_skills=missing_skills,
                improvement_suggestions=suggestions,
                salary_match=salary_match,
                location_match=location_match,
                overall_score=ats_score,
                optimized_resume=optimized_resume,
                keyword_density=keyword_density,
                error=None,
                processing_time=time.time() - start_time,
                memory_usage=self._monitor_memory_usage() - initial_memory,
                role_type=role_type,
                role_specific_metrics=role_metrics
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in resume analysis: {str(e)}")
            # Create default result with error
            result = AnalysisResult(
                ats_score=0.0,
                industry_match_score=0.0,
                experience_match_score=0.0,
                skill_matches={},
                missing_critical_skills=[],
                improvement_suggestions=[],
                salary_match=False,
                location_match=False,
                overall_score=0.0,
                optimized_resume="",
                keyword_density={},
                error=str(e),
                processing_time=time.time() - start_time,
                memory_usage=self._monitor_memory_usage() - initial_memory,
                role_type=None,
                role_specific_metrics={}
            )
            self.logger.error(f"Error in resume analysis: {str(e)}")
            return result
        """
        Analyze resume against job description with industry-specific scoring.
        
        This method implements a memory-efficient analysis pipeline with:
        1. Progressive text chunking for large documents
        2. Batch processing of embeddings
        3. Automatic memory management
        4. Comprehensive error handling
        
        Args:
            resume_text: The resume text to analyze
            job_description: The job description to match against
            target_job: Optional specific job title to target
            target_industry: Optional specific industry to target
            
        Returns:
            AnalysisResult: Detailed analysis results
            
        Raises:
            RuntimeError: If analysis fails due to system constraints
            ValueError: If input validation fails
        """
        try:
            self.logger.info("Starting resume analysis")
            with self.monitor.track_performance('analyze_resume'):
                # Check resource limits
                resource_status = self.monitor.check_resource_limits()
                if not resource_status['healthy']:
                    for warning in resource_status['warnings']:
                        self.monitor.log_warning('analyze_resume', warning)
                    for critical in resource_status['critical']:
                        self.monitor.log_error('analyze_resume', critical, 'P1')
                        if 'Memory usage critical' in critical:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                
                # Process resume text in chunks if too long
                max_length = 512  # Max length for transformer
                if len(resume_text.split()) > max_length:
                    self.logger.info("Long resume detected, processing in chunks")
                    resume_chunks = self._split_text_into_chunks(resume_text, max_length)
                    resume_embedding = self._process_text_chunks(resume_chunks)
                else:
                    resume_embedding = self._get_text_embedding(resume_text)
                
                # Find matching job requirements
                job_req = self._find_matching_requirements(
                    job_description, target_job, target_industry
                )
                
                if not job_req:
                    raise ValueError("Could not match job description to known patterns")
                
                # Calculate skill matches using transformer embeddings
                required_skills = job_req.required_skills
                skill_matches = {}
                for skill in required_skills:
                    skill_embedding = self._get_text_embedding(skill)
                    similarity = F.cosine_similarity(
                        resume_embedding, 
                        skill_embedding, 
                        dim=1
                    ).mean().item()
                    # Only store the original version
                    skill_matches[skill] = float(similarity)
                
                # Calculate other scores
                experience_score = self._calculate_experience_match(resume_text, job_req)
                industry_score = self._calculate_industry_match(resume_embedding, job_req)
                location_match = self._check_location_match(resume_text, job_req)
                salary_match = self._check_salary_match(resume_text, job_req)
                
                # Calculate overall ATS score
                ats_score = self._calculate_ats_score(
                    skill_matches, experience_score, industry_score,
                    location_match, salary_match
                )
                
                # Calculate keyword density using CX and CS patterns from training data
                keyword_density = {}
                
                # Build patterns dynamically from role_patterns for accurate role detection
                role_type = None
                role_scores = {'cx': 0, 'cs': 0}
                
                # First determine the role type
                for role in ['cx', 'cs']:
                    # Combine all patterns for the role
                    all_terms = (
                        self.role_patterns[role]['keywords'] +
                        self.role_patterns[role]['skills'] +
                        self.role_patterns[role]['tools'] +
                        self.role_patterns[role]['metrics']
                    )
                    # Count matches for role determination
                    role_scores[role] = sum(1 for term in all_terms if term.lower() in resume_lower)
                
                # Set role type based on highest score
                role_type = max(role_scores.items(), key=lambda x: x[1])[0]
                self.logger.info(f'Detected role type: {role_type} (scores: {role_scores})')
                
                # Build role-specific patterns
                role_patterns = {
                    'keywords': self.role_patterns[role_type]['keywords'],
                    'skills': self.role_patterns[role_type]['skills'],
                    'tools': self.role_patterns[role_type]['tools'],
                    'metrics': self.role_patterns[role_type]['metrics'],
                    'achievements': [
                        r'(?:improved|increased|reduced|decreased|managed|led|developed|implemented|'
                        r'launched|established|built|designed|transformed|enhanced|optimized|created|'
                        r'delivered|achieved|executed|streamlined|automated|scaled|grew|expanded)'
                        r'\s+(?:[^.]*?(?:customer|client|user|revenue|retention|satisfaction)[^.]*)'
                    ]
                }
                
                # Compile patterns for each category
                compiled_patterns = {}
                for category, terms in role_patterns.items():
                    if category == 'achievements':
                        pattern = '|'.join(terms)  # achievements is already a regex pattern
                    else:
                        # Escape special characters and join terms
                        pattern = '|'.join(re.escape(term) for term in terms)
                    try:
                        compiled_patterns[category] = re.compile(pattern, re.IGNORECASE)
                        self.logger.debug(f'Compiled pattern for {category}')
                    except Exception as e:
                        self.logger.error(f'Error compiling pattern for {category}: {str(e)}')
                
                # Input validation
                if not resume_text or not resume_text.strip():
                    self.logger.error("Empty resume text provided")
                    return keyword_density
                
                try:
                    # Normalize text once
                    resume_lower = resume_text.lower()
                    words = word_tokenize(resume_lower)
                    total_words = len(words)
                    
                    if total_words == 0:
                        self.logger.error("Resume contains no words after tokenization")
                        return keyword_density
                    
                    # Pre-compile patterns for each category
                    compiled_patterns = {}
                    for category, pattern in role_patterns.items():
                        try:
                            compiled_patterns[category] = re.compile(pattern, re.IGNORECASE)
                            self.logger.debug(f"Compiled pattern for {category}")
                        except Exception as e:
                            self.logger.error(f"Error compiling pattern for {category}: {str(e)}")
                    
                    # Calculate densities for each category
                    for category, pattern in compiled_patterns.items():
                        try:
                            matches = pattern.findall(resume_lower)
                            count = len(matches)
                            
                            # For achievements, we want to count each achievement separately
                            if category == 'achievements':
                                for match in matches:
                                    key = f"{category}:{match[:50]}..." if len(match) > 50 else f"{category}:{match}"
                                    keyword_density[key] = 1  # Each achievement counts as one occurrence
                            else:
                                # For other categories, calculate density as before
                                density = (count / total_words) * 100
                                for match in set(matches):
                                    key = f"{category}:{match}"
                                    keyword_density[key] = round(density, 1)
                            
                            # Detailed logging with category-specific details
                            log_msg = f"Category: {category} | Matches: {matches} | Count: {count}"
                            if category != 'achievements':
                                log_msg += f" | Density: {density:.1f}%"
                            log_msg += f" | Pattern: {pattern.pattern}"
                            self.logger.debug(log_msg)
                            
                            if count == 0:
                                self.logger.warning(f"No matches found for {category}")
                                
                        except Exception as e:
                            self.logger.error(
                                f"Error calculating density for {category}: {str(e)}", 
                                exc_info=True
                            )
                            
                except Exception as e:
                    self.logger.error(
                        f"Critical error in keyword density calculation: {str(e)}", 
                        exc_info=True
                    )
                    return keyword_density
                
                return keyword_density
                
                # Generate improvement suggestions
                missing_skills = self._identify_missing_skills(skill_matches)
                suggestions = self._generate_suggestions(
                    skill_matches, experience_score, industry_score,
                    location_match, salary_match, job_req
                )
                
                # Generate optimized resume with role-specific optimization
                optimized_resume = self._optimize_resume(
                    resume_text,
                    missing_skills,
                    suggestions,
                    skill_matches,
                    role_type
                )
                
                self.logger.info("Resume analysis completed successfully")
                return AnalysisResult(
                    ats_score=ats_score,
                    industry_match_score=industry_score,
                    experience_match_score=experience_score,
                    skill_matches=skill_matches,
                    missing_critical_skills=missing_skills,
                    improvement_suggestions=suggestions,
                    salary_match=salary_match,
                    location_match=location_match,
                    overall_score=ats_score,
                    optimized_resume=optimized_resume,
                    keyword_density=keyword_density
                )
                
        except Exception as e:
            error_msg = f"Resume analysis failed: {str(e)}"
            self.monitor.log_error('analyze_resume', error_msg, 'P1')
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _find_matching_requirements(self, 
                                  job_description: str,
                                  target_job: Optional[str] = None,
                                  target_industry: Optional[str] = None) -> Optional[JobRequirements]:
        """Find matching requirements using advanced NLP techniques."""
        try:
            # Preprocess job description with enhanced logging
            self.logger.debug("Starting job description preprocessing")
            processed_job = self.text_preprocessor.preprocess(job_description)
            
            # Extract structured information
            job_entities = processed_job['entities']
            job_noun_phrases = processed_job['noun_phrases']
            job_technical_terms = processed_job['technical_terms']
            
            self.logger.debug(f"Extracted {len(job_entities)} entities, {len(job_noun_phrases)} noun phrases, {len(job_technical_terms)} technical terms")
            
            # Determine role type using zero-shot classification
            role_labels = ['customer experience', 'customer success']
            role_results = self.zero_shot(
                job_description,
                candidate_labels=role_labels,
                hypothesis_template='This is a {} position.'
            )
            
            # Map classification results to role types
            role_type = 'cx' if role_results['labels'][0] == 'customer experience' else 'cs'
            self.logger.info(f'Role type determined by zero-shot: {role_type}')
            
            # Get role-specific patterns
            patterns = self.role_patterns[role_type]
            
            # Use semantic search for skill matching
            all_skills = patterns['skills'] + patterns['tools']
            skill_matches = self.semantic_search.search(
                processed_job['clean_text'],
                all_skills,
                top_k=10
            )
            
            # Extract required skills with confidence scores
            required_skills = [
                skill for skill, score in skill_matches
                if score > 0.7  # High confidence threshold
            ]
            
            # Extract experience requirements using NER and pattern matching
            experience_info = self._extract_experience_info(job_description)
            
            # Create JobRequirements object
            job_req = JobRequirements(
                title=target_job or experience_info['job_title'],
                industry=target_industry or experience_info['industry'],
                responsibilities=[],  # Would need to extract these from job description
                qualifications=[],    # Would need to extract these from job description
                required_skills=set(required_skills),
                experience_years=int(experience_info['min_years']),
                location='',  # Would need to extract this from job description
                job_type=role_type,
                salary_range=(0.0, 0.0),  # Would need to extract this from job description
                priority_skills=set([s for s, _ in skill_matches if s not in required_skills])
            )
            
            return job_req
            
        except Exception as e:
            self.logger.error(f'Error in requirement matching: {str(e)}')
            return None
        """Find matching job requirements from the database with CX and CS pattern matching."""
        self.logger.info("Finding matching requirements with CX/CS patterns")
        
        try:
            # Define CX and CS patterns for role identification
            cx_patterns = [
                r'customer\s+experience',
                r'cx\s+',
                r'ux\s+',
                r'user\s+experience',
                r'journey\s+map',
                r'voice\s+of\s+customer',
                r'customer\s+insight',
                r'experience\s+design',
                r'service\s+design'
            ]
            
            cs_patterns = [
                r'customer\s+success',
                r'cs\s+',
                r'customer\s+support',
                r'customer\s+service',
                r'account\s+management',
                r'customer\s+relationship',
                r'customer\s+retention',
                r'customer\s+satisfaction'
            ]
            
            # Determine role type
            role_type = None
            job_desc_lower = job_description.lower()
            
            for pattern in cx_patterns:
                if re.search(pattern, job_desc_lower):
                    role_type = 'cx'
                    break
                    
            if not role_type:
                for pattern in cs_patterns:
                    if re.search(pattern, job_desc_lower):
                        role_type = 'cs'
                        break
            
            if role_type:
                self.logger.info(f"Detected role type: {role_type.upper()}")
            
            # Try exact match first if target_job provided
            if target_job and target_job in self.data_manager.job_requirements:
                req = self.data_manager.job_requirements[target_job]
                if not target_industry or req.industry == target_industry:
                    self.logger.info(f"Found exact match for {target_job}")
                    return req
            
            # Try semantic matching with role type consideration
            best_match = None
            best_score = 0.0
            
            # Encode job description
            encoded_job = self._get_text_embedding(job_description.lower())
            
            # Create role-specific default if no matches found
            if not self.data_manager.job_requirements:
                self.logger.warning("No job requirements found, creating role-specific default")
                
                if role_type == 'cx':
                    default_req = JobRequirements(
                        title="Customer Experience Manager",
                        industry="Technology",
                        responsibilities=[
                            "Customer journey mapping",
                            "Voice of customer programs",
                            "Experience design",
                            "Customer insights analysis"
                        ],
                        qualifications=[
                            "CX experience",
                            "Journey mapping",
                            "Analytics experience",
                            "Bachelor's degree"
                        ],
                        required_skills={
                            "Journey Mapping",
                            "Voice of Customer",
                            "Customer Analytics",
                            "Experience Design"
                        },
                        experience_years=5,
                        location="Remote",
                        job_type="Full-time",
                        salary_range=(80000, 150000)
                    )
                else:  # CS default
                    default_req = JobRequirements(
                        title="Customer Success Manager",
                        industry="Technology",
                        responsibilities=[
                            "Customer relationship management",
                            "Account management",
                            "Customer onboarding",
                            "Retention strategies"
                        ],
                        qualifications=[
                            "CS experience",
                            "Account management",
                            "Customer support",
                            "Bachelor's degree"
                        ],
                        required_skills={
                            "Customer Success",
                            "Account Management",
                            "Customer Support",
                            "Relationship Management"
                        },
                        experience_years=3,
                        location="Remote",
                        job_type="Full-time",
                        salary_range=(70000, 130000)
                    )
                return default_req
            
            # Find best semantic match considering role type
            for job_title, req in self.data_manager.job_requirements.items():
                if target_industry and req.industry != target_industry:
                    continue
                
                # Calculate similarity score using transformers
                req_text = f"{job_title} {' '.join(req.responsibilities)} {' '.join(req.qualifications)}"
                encoded_req = self._get_text_embedding(req_text.lower())
                
                # Calculate cosine similarity between embeddings
                score = torch.nn.functional.cosine_similarity(encoded_job, encoded_req, dim=1).mean().item()
                
                # Boost score for matching role type
                if role_type == 'cx' and any(re.search(p, job_title.lower()) for p in cx_patterns):
                    score *= 1.2  # 20% boost for matching CX role
                elif role_type == 'cs' and any(re.search(p, job_title.lower()) for p in cs_patterns):
                    score *= 1.2  # 20% boost for matching CS role
                
                if score > best_score:
                    best_score = score
                    best_match = req
            
            if best_match:
                self.logger.info(f"Found best matching requirement with score {best_score:.2f}")
            else:
                self.logger.warning("No matching requirements found")
            
            return best_match
            
        except Exception as e:
            error_msg = f"Error finding matching requirements: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    def _extract_experience_info(self, text: str) -> Dict[str, any]:
        """Extract comprehensive experience information using NLP."""
        try:
            # Process text
            doc = self.nlp(text)
            
            # Initialize results
            info = {
                'min_years': 0.0,
                'max_years': 0.0,
                'industry': None,
                'job_title': None,
                'level': None
            }
            
            # Extract years of experience using patterns and NER
            year_patterns = [
                r'(\d+)\+?\s*(?:-\s*\d+)?\s*years?(?:\s+of)?\s+(?:experience|exp)',
                r'minimum\s+(?:of\s+)?(\d+)\s+years?',
                r'at\s+least\s+(\d+)\s+years?',
                r'(\d+)\s*-\s*(\d+)\s+years?'
            ]
            
            for pattern in year_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    years = [float(y) for y in match.groups() if y]
                    if len(years) == 2:
                        info['min_years'] = min(years)
                        info['max_years'] = max(years)
                    else:
                        info['min_years'] = years[0]
                        info['max_years'] = years[0] * 1.5  # Estimate max
            
            # Extract industry using NER and patterns
            industry_entities = [
                ent.text for ent in doc.ents
                if ent.label_ in {'ORG', 'PRODUCT'} and
                len(ent.text.split()) > 1  # Avoid single word entities
            ]
            
            if industry_entities:
                # Use semantic search to find most relevant industry
                matches = self.semantic_search.search(
                    text,
                    industry_entities,
                    top_k=1
                )
                if matches:
                    info['industry'] = matches[0][0]
            
            # Extract job title using noun phrases and patterns
            title_patterns = [
                r'(?:senior|lead|principal|staff)\s+[\w\s]+(?:engineer|developer|architect)',
                r'(?:software|systems|solutions)\s+[\w\s]+(?:engineer|developer|architect)',
                r'(?:product|project|program)\s+[\w\s]+(?:manager|lead)',
                r'(?:customer|client)\s+[\w\s]+(?:manager|specialist|representative)'
            ]
            
            for pattern in title_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    info['job_title'] = match.group(0)
                    break
                if info['job_title']:
                    break
            
            # Determine level from job title or context
            level_indicators = {
                'senior': ['senior', 'sr', 'lead', 'principal', 'staff'],
                'mid': ['intermediate', 'mid', 'ii', '2'],
                'junior': ['junior', 'jr', 'entry', 'associate', 'i', '1']
            }
            
            for level, indicators in level_indicators.items():
                if any(ind in text.lower() for ind in indicators):
                    info['level'] = level
                    break
            
            return info
            
        except Exception as e:
            self.logger.error(f'Error extracting experience info: {str(e)}')
            return {
                'min_years': 0.0,
                'max_years': 0.0,
                'industry': None,
                'job_title': None,
                'level': None
            }
        """Extract years of experience from text using regex patterns"""
        try:
            # Common patterns for experience matching
            patterns = [
                r'\b(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience\b',  # "5+ years experience"
                r'\b(?:experience\s*:?\s*)(\d+)\+?\s*(?:years?|yrs?)\b',     # "Experience: 5 years"
                r'\((\d{4})\s*-\s*(?:present|current|now)\)',               # "(2018-present)"
                r'\((\d{4})\s*-\s*(\d{4})\)',                              # "(2018-2023)"
            ]
            
            total_years = 0.0
            current_year = 2025  # Use a fixed year for consistent testing
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) == 1:
                        if match.group(1).isdigit():
                            # Direct year specification
                            if len(match.group(1)) == 4:  # Full year
                                year = int(match.group(1))
                                total_years = max(total_years, current_year - year)
                            else:  # Number of years
                                total_years = max(total_years, float(match.group(1)))
                    elif len(match.groups()) == 2:
                        # Date range
                        start_year = int(match.group(1))
                        end_year = int(match.group(2))
                        total_years = max(total_years, end_year - start_year)
            
            return total_years
        
        except Exception as e:
            self.logger.error(f"Error extracting experience years: {str(e)}")
            return 0.0

    def _calculate_experience_match(self, resume_text: str, job_description: str) -> float:
        """Calculate experience match score.
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            
        Returns:
            float: Experience match score between 0 and 1
        """
        try:
            # Extract years from resume
            resume_experience = self._extract_years_experience(resume_text)
            
            # Extract years from job description
            required_experience = self._extract_years_experience(job_description)
            
            # Calculate match score
            if required_experience == 0:
                return 1.0
                
            if resume_experience >= required_experience:
                return 1.0
                
            match_score = min(resume_experience / required_experience, 1.0)
            self.logger.debug("Experience match score: %f (resume: %d, required: %d)", 
                            match_score, resume_experience, required_experience)
            return match_score
            
        except Exception as e:
            self.logger.error(f"Error calculating experience match: {str(e)}")
            return 0.0

    def _split_text_into_chunks(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks of max_length tokens"""
        try:
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                word_length = len(self.tokenizer.encode(word))
                if current_length + word_length > max_length:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = word_length
                else:
                    current_chunk.append(word)
                    current_length += word_length
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error splitting text into chunks: {str(e)}")
            return [text]  # Return original text as single chunk
    
    def _process_text_chunks(self, chunks: List[str], batch_size: int = 4) -> torch.Tensor:
        """Process multiple text chunks and combine their embeddings with batching.
        
        Args:
            chunks: List of text chunks to process
            batch_size: Number of chunks to process in each batch
            
        Returns:
            torch.Tensor: Combined embeddings from all chunks
        """
        try:
            embeddings = []
            total_chunks = len(chunks)
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_chunks)
                batch_chunks = chunks[start_idx:end_idx]
                
                self.logger.debug(f"Processing batch {batch_idx + 1}/{total_batches} "
                                 f"(chunks {start_idx + 1}-{end_idx}/{total_chunks})")
                
                # Check memory before processing batch
                if self._monitor_memory_usage() > 0.85:  # 85% threshold
                    self._clear_cache()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Process each chunk in the current batch
                batch_embeddings = []
                for chunk_text in batch_chunks:
                    chunk_embedding = self._get_text_embedding(chunk_text)
                    batch_embeddings.append(chunk_embedding)
                
                # Average the batch embeddings
                batch_embedding = torch.stack(batch_embeddings).mean(dim=0)
                embeddings.append(batch_embedding)
            
            # Stack and average embeddings
            stacked = torch.stack(embeddings)
            return torch.mean(stacked, dim=0)
            
        except Exception as e:
            self.logger.error(f"Error processing text chunks: {str(e)}")
            # Return embedding of first chunk as fallback
            return self._get_text_embedding(chunks[0])
    
    def _calculate_industry_match(self, resume_text: str, job_description: str) -> float:
        """Calculate industry match score using transformers"""
        try:
            # Extract industry from job description
            industry = self._extract_industry(job_description)
            if not industry:
                return 0.5  # Default score if no industry found
                
            # Get embeddings
            resume_embedding = self._get_text_embedding(resume_text.lower())
            industry_embedding = self._get_text_embedding(industry.lower())
            
            # Calculate similarity
            similarity = F.cosine_similarity(resume_embedding, industry_embedding, dim=1).mean().item()
            
            return float(min(similarity, 1.0))
            
        except Exception as e:
            self.logger.error(f"Error calculating industry match: {str(e)}")
            return 0.5  # Default score on error
            
    def _extract_industry(self, text: str) -> Optional[str]:
        """Extract industry from text using NLP."""
        try:
            doc = self.nlp(text)
            industry_keywords = [
                'technology', 'software', 'healthcare', 'finance', 'retail',
                'manufacturing', 'education', 'consulting', 'media', 'advertising'
            ]
            
            for sent in doc.sents:
                sent_text = sent.text.lower()
                if 'industry' in sent_text:
                    for keyword in industry_keywords:
                        if keyword in sent_text:
                            return keyword
            return None
        except Exception as e:
            self.logger.error(f"Error extracting industry: {str(e)}")
            return None

    def _check_location_match(self, resume_text: str, job_description: str) -> bool:
        """Check if resume location matches job location using transformers"""
        try:
            # Extract locations
            job_location = self._extract_location(job_description)
            if not job_location:
                return True  # Default to True if no location requirement
                
            resume_location = self._extract_location(resume_text)
            if not resume_location:
                return False  # No location found in resume
                
            # Get embeddings
            resume_loc_embedding = self._get_text_embedding(resume_location.lower())
            job_loc_embedding = self._get_text_embedding(job_location.lower())
            
            # Calculate similarity
            similarity = F.cosine_similarity(resume_loc_embedding, job_loc_embedding, dim=1).mean().item()
            
            # Consider it a match if similarity is above threshold
            return similarity > 0.7  # Adjust threshold as needed
            
        except Exception as e:
            self.logger.error(f"Error checking location match: {str(e)}")
            return True  # Default to True on error
            
    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location from text using NLP."""
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'LOC']:
                    return ent.text
            return None
        except Exception as e:
            self.logger.error(f"Error extracting location: {str(e)}")
            return None

    def _check_salary_match(self, resume_text: str, job_description: str) -> bool:
        """Check if resume salary expectations match job salary range"""
        try:
            # Extract salary ranges
            job_salary = self._extract_salary_range(job_description)
            if not job_salary:
                return True  # No salary requirement specified
                
            resume_salary = self._extract_salary_range(resume_text)
            if not resume_salary:
                return True  # If no salary mentioned, assume match
                
            # Parse ranges
            try:
                min_job, max_job = map(float, job_salary.split('-'))
                min_resume, max_resume = map(float, resume_salary.split('-'))
            except ValueError:
                return True  # If parsing fails, assume match
                
            # Check if ranges overlap
            return min_resume <= max_job and max_resume >= min_job
            
        except Exception as e:
            self.logger.error(f"Error checking salary match: {str(e)}")
            return True  # Default to True on error
            
    def _extract_salary_range(self, text: str) -> Optional[str]:
        """Extract salary range from text using regex."""
        try:
            # Look for patterns like $X-$Y or $X to $Y
            patterns = [
                r'\$([\d,]+)\s*-\s*\$([\d,]+)',
                r'\$([\d,]+)\s*to\s*\$([\d,]+)',
                r'([\d,]+)k\s*-\s*([\d,]+)k'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    min_sal = float(match.group(1).replace(',', ''))
                    max_sal = float(match.group(2).replace(',', ''))
                    if 'k' in pattern:
                        min_sal *= 1000
                        max_sal *= 1000
                    return f"{min_sal}-{max_sal}"
            return None
        except Exception as e:
            self.logger.error(f"Error extracting salary range: {str(e)}")
            return None
        
    def _initialize_device(self) -> torch.device:
        """Initialize device with advanced memory management.
        
        This method implements a sophisticated device initialization strategy:
        1. Detects available compute devices (CPU, MPS, CUDA)
        2. Configures device-specific memory limits
        3. Implements fallback mechanisms
        4. Sets up device-specific optimizations
        5. Validates device stability
        
        Returns:
            torch.device: Optimal device for current system
            
        Note:
            Prioritizes stability over performance for production use
        """
        try:
            # Check CUDA availability first
            if torch.cuda.is_available():
                device = torch.device('cuda')
                # Configure CUDA specific settings
                torch.cuda.set_per_process_memory_fraction(0.7)  # Reserve 30% for system
                torch.cuda.empty_cache()
                self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
                return device
            
            # Check MPS availability (Apple Silicon)
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                # MPS specific optimizations
                if hasattr(torch.mps, 'set_per_process_memory_fraction'):
                    torch.mps.set_per_process_memory_fraction(0.7)
                self.logger.info("Using MPS device (Apple Silicon)")
                return device
            
            # Fallback to CPU with optimizations
            device = torch.device('cpu')
            torch.set_num_threads(max(1, os.cpu_count() // 2))  # Use half available cores
            self.logger.info("Using CPU device with thread optimization")
            return device
            
        except Exception as e:
            self.logger.error(f"Error initializing device: {str(e)}")
            self.logger.warning("Falling back to CPU device")
            return torch.device('cpu')
        try:
            if torch.backends.mps.is_available():
                # Set conservative memory limit (70% of available memory)
                try:
                    torch.mps.set_per_process_memory_fraction(0.7)
                    self.logger.info("Set MPS memory limit to 70%")
                except Exception as e:
                    self.logger.warning(f"Failed to set MPS memory limit: {e}")
                
                # Basic MPS configuration
                device = torch.device("mps")
                self.logger.info("Using MPS for hardware acceleration")
                return device
            else:
                self.logger.info("MPS not available, using CPU")
                return torch.device("cpu")
                
        except Exception as e:
            self.logger.error(f"Error in device initialization: {e}")
            return torch.device("cpu")

    def _clear_cache(self) -> None:
        """Clear device cache and optimize memory usage.
        
        This method handles cache clearing with several strategies:
        1. Checks current memory usage against thresholds
        2. Clears appropriate cache based on device type
        3. Attempts to free unused memory
        4. Logs memory usage before and after clearing
        
        Note:
            This is called automatically when memory usage exceeds thresholds,
            but can also be called manually if needed.
        """
        try:
            # Get initial memory state
            initial_memory = psutil.Process().memory_info().rss
            initial_percent = (initial_memory / psutil.virtual_memory().total) * 100
            
            self.logger.debug(
                f"Initial memory usage: {initial_percent:.1f}% "
                f"({initial_memory / (1024 * 1024):.1f} MB)"
            )
            
            # Clear appropriate cache based on device
            if self.device.type == 'mps':
                # MPS-specific optimizations
                torch.mps.empty_cache()
                self.logger.info("Cleared MPS cache")
                
                # Additional MPS memory management
                if hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
                    self.logger.debug("Synchronized MPS device")
            
            # Safe to call these regardless of device
            torch.cuda.empty_cache()
            
            # Get final memory state
            final_memory = psutil.Process().memory_info().rss
            final_percent = (final_memory / psutil.virtual_memory().total) * 100
            
            memory_freed = initial_memory - final_memory
            if memory_freed > 0:
                self.logger.info(
                    f"Freed {memory_freed / (1024 * 1024):.1f} MB of memory. "
                    f"Current usage: {final_percent:.1f}%"
                )
            else:
                self.logger.warning(
                    "No memory freed by cache clearing. "
                    f"Current usage: {final_percent:.1f}%"
                )
                
        except Exception as e:
            self.logger.error(
                f"Error during cache clearing: {str(e)}. "
                "Memory state may be inconsistent."
            )
            
    def _monitor_memory_usage(self) -> float:
        """Monitor system memory usage with advanced metrics.
        
        This method provides comprehensive memory monitoring:
        1. Tracks system memory usage
        2. Monitors device-specific memory (GPU/MPS)
        3. Tracks process-specific memory
        4. Implements warning thresholds
        5. Provides detailed logging
        
        Returns:
            float: Current memory usage as a percentage
        
        Note:
            Implements memory monitoring based on device type and system state
        """
        try:
            # Get system memory metrics
            vm = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info().rss
            
            # Calculate usage percentages
            system_usage = vm.percent / 100.0
            process_usage = process_memory / vm.total
            
            # Get device-specific memory metrics
            device_memory = 0.0
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                device_memory = torch.cuda.memory_allocated(device) / torch.cuda.max_memory_allocated(device)
            elif hasattr(torch.mps, 'current_allocated_memory'):
                device_memory = torch.mps.current_allocated_memory() / vm.total
            
            # Calculate weighted usage
            total_usage = max(system_usage, process_usage, device_memory)
            
            # Log detailed metrics
            self.logger.debug(
                f"Memory Usage - System: {system_usage:.1%}, "
                f"Process: {process_usage:.1%}, Device: {device_memory:.1%}"
            )
            
            # Check warning thresholds
            if total_usage > 0.85:  # 85% critical
                self.logger.warning(f"Critical memory usage: {total_usage:.1%}")
            elif total_usage > 0.7:  # 70% warning
                self.logger.info(f"High memory usage: {total_usage:.1%}")
            
            return total_usage
            
        except Exception as e:
            self.logger.error(f"Error monitoring memory: {str(e)}")
            return 1.0  # Return high usage to trigger conservative behavior
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Log detailed memory information
            self.logger.debug(
                f"Memory usage: {memory_percent:.1f}% "
                f"({memory_info.rss / (1024 * 1024):.1f} MB)"
            )
            
            # Check against thresholds
            if memory_percent > 85.0:  # Critical threshold
                self.logger.critical(
                    f"Critical memory usage: {memory_percent:.1f}%"
                )
                if self.device.type == 'mps':
                    torch.mps.empty_cache()
                    self.logger.info("Cleared MPS cache")
            elif memory_percent > 70.0:  # Warning threshold
                self.logger.warning(
                    f"High memory usage: {memory_percent:.1f}%"
                )
                
            return memory_percent
            
        except Exception as e:
            self.logger.error(f"Error monitoring memory: {str(e)}")
            return 0.0

    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Generate text embeddings with essential error recovery.
        
        Implements core error handling and memory management for
        production stability.
        
        Args:
            text: Input text to embed
            
        Returns:
            torch.Tensor: Generated embedding or zero tensor on failure
        """
        max_retries = 2
        current_try = 0
        
        while current_try <= max_retries:
            try:
                # Basic tokenization with fixed length
                inputs = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # Core inference with memory safety
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = self._mean_pooling(
                        outputs.last_hidden_state,
                        inputs['attention_mask']
                    )
                return embeddings
                
            except RuntimeError as e:
                current_try += 1
                error_msg = str(e)
                
                if "out of memory" in error_msg:
                    self.logger.warning(f"OOM error (attempt {current_try}): {error_msg}")
                    if self.device.type == "mps":
                        torch.mps.empty_cache()
                    if current_try <= max_retries:
                        continue
                        
                self.logger.error(f"Model inference failed: {error_msg}")
                if current_try <= max_retries:
                    continue
                    
            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}")
                break
                
        # Fallback to zero tensor after all retries
        return torch.zeros(
            1,
            self.model.config.hidden_size,
            device=self.device
        )
    def _mean_pooling(self, token_embeddings: torch.Tensor, 
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """Calculate mean pooling with attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _calculate_title_match(self, resume_text: str, job_title: str) -> float:
        """Calculate how well the resume title matches the job title.
        
        Args:
            resume_text: The resume text to analyze
            job_title: The job title to match against
            
        Returns:
            float: Title match score between 0 and 1
        """
        # Extract current/recent title from resume
        resume_lines = resume_text.split('\n')
        resume_title = ''
        for line in resume_lines[:10]:  # Check first 10 lines for title
            if any(word.lower() in line.lower() for word in job_title.lower().split()):
                resume_title = line
                break
        
        if not resume_title:
            return 0.0
            
        # Generate embeddings for semantic comparison
        title_embedding = self._get_text_embedding(job_title)
        resume_title_embedding = self._get_text_embedding(resume_title)
        
        if title_embedding is None or resume_title_embedding is None:
            return 0.0
            
        # Calculate cosine similarity
        similarity = F.cosine_similarity(title_embedding, resume_title_embedding, dim=1)
        return float(similarity.item()) if similarity > self.similarity_threshold else 0.0

    def _calculate_keyword_density(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword density percentage.
        
        Args:
            text: Text to analyze
            keywords: Keywords to check density for
            
        Returns:
            Density percentage (0-100)
        """
        words = text.lower().split()
        total_words = len(words)
        
        if not total_words:
            return 0
            
        keyword_count = sum(1 for word in words 
                           if any(k.lower() in word for k in keywords))
        
        return (keyword_count / total_words) * 100
        """Calculate keyword density in text."""
        words = text.lower().split()
        total_words = len(words)
        density = {}
        
        for keyword in keywords:
            count = text.lower().count(keyword.lower())
            density[keyword] = count / total_words if total_words > 0 else 0
            
        return density

    def _calculate_ats_score(self,
                           skill_matches: Dict[str, float],
                           experience_score: float,
                           industry_score: float,
                           location_match: bool,
                           salary_match: bool,
                           title_match: float = 0.0,
                           role_metrics: Optional[Dict[str, int]] = None) -> float:
        """Calculate overall ATS score with role-specific metrics.
        
        This method calculates a comprehensive ATS score by combining:
        1. Standard metrics (skills, experience, industry, location, salary)
        2. Role-specific metrics for CX/CS positions when available
        
        Args:
            skill_matches: Dictionary of skill matches and their scores
            experience_score: Score for experience match
            industry_score: Score for industry match
            location_match: Whether location matches
            salary_match: Whether salary expectations match
            role_metrics: Optional role-specific metrics
            
        Returns:
            float: Overall ATS score between 0 and 1
        """
        # Calculate skill score from match_score or overall score
        if isinstance(skill_matches, dict):
            if 'match_score' in skill_matches:
                skill_score = float(skill_matches['match_score'])
            else:
                # Filter out non-numeric values
                numeric_scores = {k: v for k, v in skill_matches.items() 
                                if isinstance(v, (int, float))}
                skill_score = sum(numeric_scores.values()) / len(numeric_scores) if numeric_scores else 0.0
        else:
            skill_score = 0.0
        
        # Calculate base score with standard weights including title match
        base_score = (
            skill_score * self.weights['skills'] +
            experience_score * self.weights['experience'] +
            title_match * self.weights['title'] +
            industry_score * self.weights['industry'] +
            float(location_match) * self.weights['location'] +
            float(salary_match) * self.weights['salary']
        )
        
        # Incorporate role-specific metrics if available
        if role_metrics and 'overall_alignment' in role_metrics:
            # Convert role metrics to 0-1 scale and weight them
            role_alignment = role_metrics['overall_alignment'] / 100.0
            
            # Additional role-specific bonuses
            role_bonus = 0.0
            if 'certifications' in role_metrics:  # CX role bonus
                role_bonus += (role_metrics['certifications'] / 100.0) * 0.1  # 10% weight
            if 'revenue_impact' in role_metrics:  # CS role bonus
                role_bonus += (role_metrics['revenue_impact'] / 100.0) * 0.1  # 10% weight
            
            # Combine scores with role alignment having 20% weight
            final_score = (base_score * 0.7) + (role_alignment * 0.2) + role_bonus
        else:
            final_score = base_score
        
        return min(final_score, 1.0)  # Normalize to [0, 1]

    def _identify_missing_skills(self, 
                               skill_matches: Dict[str, Any]) -> List[str]:
        """Identify critical missing skills based on skill matches dictionary.
        
        Args:
            skill_matches: Dictionary containing skill match information
            
        Returns:
            List of missing skills
        """
        if 'missing_skills' in skill_matches:
            return skill_matches['missing_skills']
            
        missing_skills = []
        for skill, score in skill_matches.items():
            if isinstance(score, (int, float)) and score < 0.5:
                missing_skills.append(skill)
        return sorted(missing_skills, key=lambda x: skill_matches.get(x, 0) if isinstance(skill_matches.get(x), (int, float)) else 0)

    def _extract_keywords(self, job_description: str) -> List[str]:
        """Extract important keywords from job description.
        
        Args:
            job_description: Job posting text
            
        Returns:
            List of important keywords
        """
        doc = self.nlp(job_description)
        keywords = []
        
        # Get nouns and noun phrases
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2:
                keywords.append(token.text)
                
        # Remove duplicates and sort
        return sorted(list(set(keywords)))
        
    def _add_keyword(self, text: str, keyword: str, count: int) -> str:
        """Add keyword to text naturally.
        
        Args:
            text: Text to modify
            keyword: Keyword to add
            count: Number of times to add
            
        Returns:
            Modified text
        """
        sentences = text.split('.')
        
        # Find sentences without the keyword
        available = [i for i, s in enumerate(sentences) 
                    if keyword.lower() not in s.lower()]
                    
        # Add keyword to random sentences
        for _ in range(min(count, len(available))):
            idx = random.choice(available)
            sentences[idx] = f"{sentences[idx].strip()} including {keyword}"
            available.remove(idx)
            
        return '. '.join(sentences)
        
    def _reduce_keyword(self, text: str, keyword: str, count: int) -> str:
        """Reduce keyword occurrences.
        
        Args:
            text: Text to modify
            keyword: Keyword to reduce
            count: Number to remove
            
        Returns:
            Modified text
        """
        words = text.split()
        indices = [i for i, word in enumerate(words) 
                  if keyword.lower() in word.lower()]
                  
        # Remove random occurrences
        for _ in range(min(count, len(indices))):
            idx = random.choice(indices)
            words[idx] = ''
            indices.remove(idx)
            
        return ' '.join(w for w in words if w)
        
    def _optimize_resume(self, text: str, keywords: List[str]) -> str:
        """Optimize resume to hit target metrics.
        
        Args:
            text: Original resume text
            keywords: Keywords to optimize for
            
        Returns:
            Optimized resume text
        """
        words = text.split()
        total_words = len(words)
        
        # Calculate current keyword occurrences
        keyword_counts = {k: text.lower().count(k.lower()) for k in keywords}
        
        # Calculate target counts for 2-4.5% density
        target_min = int(total_words * 0.02)  # 2%
        target_max = int(total_words * 0.045)  # 4.5%
        
        # Adjust keyword frequency to hit targets
        optimized = text
        for keyword, count in keyword_counts.items():
            if count < target_min:
                # Add keyword naturally
                if 'skills' in text.lower():
                    optimized += f", {keyword}"
                elif 'experience' in text.lower():
                    # Add to relevant experience points
                    lines = optimized.split('\n')
                    for i, line in enumerate(lines):
                        if keyword.lower() in line.lower():
                            continue
                        if line.strip().startswith('•'):
                            lines[i] = line.replace('.', f", utilizing {keyword}.")
                            break
                    optimized = '\n'.join(lines)
                
            elif count > target_max:
                # Remove excess occurrences
                optimized = self._reduce_keyword(optimized, keyword, count - target_max)
                
        return optimized
        """Optimize resume to achieve 90%+ ATS match and 2-4.5% keyword density.
        
        Args:
            resume_text: Original resume text
            job_description: Target job description
            original_metrics: Original ATS and density metrics
            
        Returns:
            Tuple of (optimized resume text, new metrics)
        """
        # Extract key terms from job description
        job_doc = self.nlp(job_description)
        key_terms = [
            token.text.lower() for token in job_doc 
            if not token.is_stop and not token.is_punct
        ]
        
        # Get sections from master resume
        sections = self._extract_resume_sections(resume_text)
        
        # Optimize each section while maintaining natural flow
        optimized_sections = {}
        for section_name, content in sections.items():
            # Identify relevant content for this job
            relevance_score = self._calculate_section_relevance(
                content, job_description
            )
            
            if relevance_score > 0.7:  # Keep highly relevant sections
                optimized_content = self._optimize_section_content(
                    content, key_terms
                )
                optimized_sections[section_name] = optimized_content
        
        # Combine sections into optimized resume
        optimized_text = self._combine_sections(optimized_sections)
        
        # Calculate new metrics
        new_metrics = {
            'ats': self._calculate_ats_score(
                skill_matches=self._extract_skills(optimized_text),
                experience_score=self._calculate_experience_match(optimized_text, job_req),
                industry_score=self._calculate_industry_match(
                    self._get_text_embedding(optimized_text),
                    job_req
                ),
                location_match=True,  # Preserved from original
                salary_match=True,    # Preserved from original
                title_match=self._calculate_title_match(optimized_text, job_req.title)
            ),
            'density': self._calculate_keyword_density(optimized_text, key_terms)
        }
        
        return optimized_text, new_metrics

    def _optimize_resume_text(self, resume_text: str, missing_skills: List[str], suggestions: List[str], skill_matches: Dict[str, float]) -> str:
        """Optimize resume text with enhanced keyword density.
        
        Args:
            resume_text: Original resume text
            missing_skills: List of skills to add
            suggestions: List of improvement suggestions
            skill_matches: Current skill match scores
            
        Returns:
            str: Optimized resume text with enhanced keyword density
        """
        try:
            # Split resume into sections
            sections = resume_text.split('\n\n')
            optimized_sections = []
            
            # Track keyword density
            keywords = set(skill_matches.keys()) | set(missing_skills)
            current_density = {}
            target_density = 0.02  # Target 2% density for each keyword
            
            for section in sections:
                # Calculate current keyword density
                words = section.lower().split()
                total_words = len(words)
                
                if total_words == 0:
                    optimized_sections.append(section)
                    continue
                
                for keyword in keywords:
                    keyword_count = sum(1 for word in words if keyword.lower() in word.lower())
                    current_density[keyword] = keyword_count / total_words
                
                # Optimize section
                optimized_section = section
                
                # Add missing high-priority skills
                for skill in missing_skills:
                    if current_density.get(skill, 0) < target_density:
                        # Add skill in a natural way
                        if 'skills' in section.lower():
                            optimized_section += f", {skill}"
                        elif 'experience' in section.lower():
                            # Add to relevant experience points
                            lines = optimized_section.split('\n')
                            for i, line in enumerate(lines):
                                if skill.lower() in line.lower():
                                    continue
                                if line.strip().startswith('•'):
                                    lines[i] = line.replace('.', f", utilizing {skill}.")
                                    break
                            optimized_section = '\n'.join(lines)
                
                # Enhance existing skills
                for skill, score in skill_matches.items():
                    if score < 0.7:  # Below 70% match
                        if current_density.get(skill, 0) < target_density:
                            if 'skills' in section.lower() and skill not in section:
                                optimized_section += f", {skill}"
                            elif 'experience' in section.lower():
                                lines = optimized_section.split('\n')
                                for i, line in enumerate(lines):
                                    if skill.lower() in line.lower():
                                        continue
                                    if line.strip().startswith('•'):
                                        lines[i] = line.replace('.', f", leveraging {skill}.")
                                        break
                                optimized_section = '\n'.join(lines)
                
                optimized_sections.append(optimized_section)
            
            # Combine optimized sections
            optimized = '\n\n'.join(optimized_sections)
            
            # Add or enhance role-specific sections
            role_sections = {
                'cx': ['Customer Experience', 'Support Tools', 'Service Metrics'],
                'cs': ['Customer Success', 'Account Management', 'Revenue Impact']
            }
            
            # Detect role type from keywords
            role_type = None
            if any(kw in ' '.join(keywords).lower() for kw in ['customer experience', 'cx', 'support', 'service desk']):
                role_type = 'cx'
            elif any(kw in ' '.join(keywords).lower() for kw in ['customer success', 'cs', 'account management']):
                role_type = 'cs'
            
            # Add role-specific sections if missing
            if role_type:
                for section_name in role_sections[role_type]:
                    if section_name.lower() not in optimized.lower():
                        section_content = f"\n\n{section_name}\n"
                        relevant_skills = [k for k in keywords if k.lower() in 
                            (self.CX_CS_SKILL_PATTERNS.get(f'{role_type}_tools', []) +
                             self.CX_CS_SKILL_PATTERNS.get(f'{role_type}_metrics', []) +
                             self.CX_CS_SKILL_PATTERNS.get(f'{role_type}_processes', []))]
                        if relevant_skills:
                            section_content += ", ".join(sorted(relevant_skills))
                        optimized += section_content
            
            # Add skills section if still missing
            if 'skills' not in optimized.lower():
                skills_section = "\n\nTechnical Skills\n" + ", ".join(sorted(keywords))
                optimized += skills_section
            
            # Calculate final keyword density
            final_words = optimized.lower().split()
            total_words = len(final_words)
            final_density = {}
            
            for keyword in keywords:
                keyword_count = sum(1 for word in final_words if keyword.lower() in word.lower())
                final_density[keyword] = (keyword_count / total_words) * 100  # Convert to percentage
            
            # Add density report at the end
            density_report = "\n\nKeyword Density Analysis:\n"
            for keyword, density in sorted(final_density.items(), key=lambda x: x[1], reverse=True):
                density_report += f"{keyword}: {density:.1f}%\n"
            
            optimized += density_report
            
            # Add improvement suggestions as comments
            if suggestions:
                optimized += "\n\nATS Optimization Suggestions:\n"
                optimized += "\n".join(f"• {suggestion}" for suggestion in suggestions)
            
            return optimized
            
        except Exception as e:
            self.logger.error(f"Error optimizing resume: {str(e)}")
            return resume_text
            
    def _calculate_role_metrics(self, resume_text: str, job_description: str, role_type: str) -> Dict[str, int]:
        """Calculate role-specific metrics using advanced NLP techniques.
        
        This method analyzes the resume and job description to calculate metrics specific to
        CX (Customer Experience) and CS (Customer Success) roles using NLP techniques.
        
        Args:
            resume_text: The resume text to analyze
            job_description: The job description text
            role_type: The type of role ('cx' or 'cs')
            
        Returns:
            Dict[str, int]: Role-specific metrics with scores
        """
        try:
            metrics = {}
            role_type = role_type.lower()
            
            # Get role-specific patterns and keywords
            if role_type == 'cx':
                patterns = {
                    'customer_interaction': r'(?i)(customer|client)\s+(interaction|engagement|communication|support)',
                    'cx_tools': r'(?i)(zendesk|intercom|freshdesk|helpscout|salesforce|crm)',
                    'cx_metrics': r'(?i)(csat|nps|customer satisfaction|response time|resolution time)',
                    'cx_processes': r'(?i)(escalation|ticket management|queue management|customer feedback)',
                    'cx_skills': r'(?i)(empathy|problem.solving|communication|multi.tasking|de.escalation)'
                }
            else:  # cs role
                patterns = {
                    'cs_strategy': r'(?i)(customer success|retention|expansion|upsell|growth)',
                    'cs_tools': r'(?i)(gainsight|totango|planhat|salesforce|tableau|looker)',
                    'cs_metrics': r'(?i)(churn|mrr|arr|expansion|retention|ltv|roi)',
                    'cs_processes': r'(?i)(onboarding|quarterly.review|business.review|success.plan)',
                    'cs_skills': r'(?i)(relationship.building|strategic.planning|project.management|consulting)'
                }
            
            # Process resume and job description
            combined_text = f"{resume_text}\n{job_description}"
            
            # Calculate metrics
            for metric_name, pattern in patterns.items():
                # Find all matches in resume
                resume_matches = len(re.findall(pattern, resume_text))
                
                # Find all matches in job description
                job_matches = len(re.findall(pattern, job_description))
                
                # Calculate weighted score based on matches
                if job_matches > 0:
                    # Normalize score based on job requirements
                    score = min(100, int((resume_matches / job_matches) * 100))
                else:
                    # If not mentioned in job, use absolute scale
                    score = min(100, resume_matches * 20)  # 20 points per mention up to 100
                
                metrics[metric_name] = score
            
            # Calculate overall role alignment
            metrics['overall_alignment'] = int(sum(metrics.values()) / len(metrics))
            
            # Add role-specific bonus metrics
            if role_type == 'cx':
                # Check for customer service certifications
                cert_pattern = r'(?i)(customer service|cx|support|hdmi|itil)\s+certification'
                certs = len(re.findall(cert_pattern, resume_text))
                metrics['certifications'] = min(100, certs * 25)  # 25 points per cert up to 100
            else:  # cs role
                # Check for revenue impact mentions
                revenue_pattern = r'(?i)(increased|grew|improved)\s+\$(\d+[km]?|\d+%|revenue|arr|mrr)'
                revenue_impacts = len(re.findall(revenue_pattern, resume_text))
                metrics['revenue_impact'] = min(100, revenue_impacts * 20)  # 20 points per mention up to 100
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating role metrics: {str(e)}")
            return {'error': 0, 'overall_alignment': 0}
    
    def _generate_suggestions(self,
                            skill_matches: Dict[str, float],
                            experience_score: float,
                            industry_score: float,
                            location_match: bool,
                            salary_match: bool,
                            job_req: JobRequirements,
                            role_metrics: Optional[Dict[str, int]] = None) -> List[str]:
        """Generate improvement suggestions based on analysis results including role-specific metrics.
        
        This method generates tailored suggestions by analyzing:
        1. Standard metrics (skills, experience, industry, etc.)
        2. Role-specific metrics for CX/CS positions
        3. Missing critical components for each role type
        
        Args:
            skill_matches: Dictionary of skill matches and scores
            experience_score: Score for experience match
            industry_score: Score for industry match
            location_match: Whether location matches
            salary_match: Whether salary expectations match
            job_req: Job requirements object
            role_metrics: Optional role-specific metrics
            
        Returns:
            List[str]: Prioritized improvement suggestions
        """
        suggestions = []
        
        # Skill suggestions based on TF-IDF analysis
        missing_skills = self._identify_missing_skills(skill_matches)
        if missing_skills:
            top_missing = sorted(missing_skills, key=lambda x: skill_matches.get(x, 0))[:5]
            suggestions.append(
                f"Add these key skills: {', '.join(top_missing)}"
            )
        
        # Experience suggestions
        if experience_score < 0.8:
            suggestions.append(
                f"Highlight relevant experience matching the {job_req.experience_years}+ "
                "years requirement. Consider quantifying achievements."
            )
        
        # Industry match suggestions
        if industry_score < 0.7:
            suggestions.append(
                f"Add more {job_req.industry}-specific terminology and achievements. "
                "Focus on relevant industry projects and outcomes."
            )
        
        # Location suggestions
        if not location_match:
            suggestions.append(
                f"Address location requirements for {job_req.location}. "
                "Consider mentioning relocation willingness or remote work preferences."
            )
        
        # Salary suggestions
        if not salary_match and job_req.salary_range and job_req.salary_range != (0.0, 0.0):
            min_salary, max_salary = job_req.salary_range
            suggestions.append(
                f"Consider aligning salary expectations with role range "
                f"(${min_salary/1000:.0f}k - ${max_salary/1000:.0f}k)"
            )
        
        # Add role-specific suggestions if metrics available
        if role_metrics:
            # Common metrics to check for both roles
            metric_thresholds = {
                'customer_interaction': 70,
                'cx_tools': 60,
                'cx_metrics': 60,
                'cx_processes': 70,
                'cx_skills': 70,
                'cs_strategy': 70,
                'cs_tools': 60,
                'cs_metrics': 60,
                'cs_processes': 70,
                'cs_skills': 70
            }
            
            # Add suggestions for low-scoring metrics
            for metric, score in role_metrics.items():
                if metric == 'error' or metric == 'overall_alignment':
                    continue
                    
                threshold = metric_thresholds.get(metric, 70)
                if score < threshold:
                    if 'tools' in metric:
                        suggestions.append(
                            f"Strengthen your {metric.replace('_', ' ')} experience. "
                            "Add specific examples of using relevant software."
                        )
                    elif 'metrics' in metric:
                        suggestions.append(
                            f"Highlight your experience with {metric.replace('_', ' ')}. "
                            "Include specific KPIs and achievements."
                        )
                    elif 'processes' in metric:
                        suggestions.append(
                            f"Elaborate on your {metric.replace('_', ' ')} experience. "
                            "Detail your role in key workflows."
                        )
                    elif 'skills' in metric:
                        suggestions.append(
                            f"Emphasize your {metric.replace('_', ' ')}. "
                            "Provide concrete examples demonstrating these abilities."
                        )
            
            # Role-specific bonus suggestions
            if 'certifications' in role_metrics and role_metrics['certifications'] < 50:
                suggestions.append(
                    "Consider adding relevant customer service certifications "
                    "(e.g., CCXP, HDI, ITIL) to strengthen your profile."
                )
            if 'revenue_impact' in role_metrics and role_metrics['revenue_impact'] < 50:
                suggestions.append(
                    "Quantify your impact on revenue metrics (ARR, MRR, retention). "
                    "Include specific growth and expansion achievements."
                )
        
        return suggestions

    def _calculate_skill_matches(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Calculate skill matches between resume and job description.
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            
        Returns:
            Dict with match scores and details
        """
        try:
            # Extract skills
            resume_skills = self._extract_skills(resume_text)
            job_skills = self._extract_skills(job_description)
            
            # Calculate matches
            matches = set(resume_skills) & set(job_skills)
            missing = set(job_skills) - set(resume_skills)
            
            # Calculate match score
            if job_skills:
                match_score = len(matches) / len(job_skills)
            else:
                match_score = 0.0
                
            result = {
                'matches': list(matches),
                'missing_skills': list(missing),
                'match_score': match_score
            }
            
            self.logger.debug("Skill match results: %s", result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error in skill matching: {str(e)}")
            return {'matches': [], 'missing_skills': [], 'match_score': 0.0}

    def health_check(self):
        """Verify critical methods are available."""
        if not hasattr(self, '_calculate_skill_matches'):
            self.logger.critical("Missing critical method: _calculate_skill_matches")
            raise RuntimeError("Missing required analysis component")
        return True
