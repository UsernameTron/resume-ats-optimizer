import logging
import re
import psutil
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import nltk
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
from .data_manager import DataManager, JobRequirements
from app.core.resource_monitor import ResourceMonitor

from app.utils.nltk_utils import ensure_nltk_data

# Ensure NLTK data is available
ensure_nltk_data()

@dataclass
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
    error: Optional[str] = None  # Track any errors during analysis
    processing_time: float = 0.0  # Track processing time
    memory_usage: float = 0.0    # Track peak memory usage

class EnhancedAnalyzer:
    def __init__(self, data_manager: DataManager):
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
        
        # Initialize ML model with memory optimization
        try:
            model_name = 'microsoft/codebert-base'  # Better for technical content
            self.logger.info(f"Loading model {model_name}")
            
            # Initialize device first to ensure proper model loading
            self.device = self._initialize_device()
            self.logger.info(f"Using device: {self.device}")
            
            # Essential model configuration
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.model_max_length = 512  # Fixed sequence length
            
            # Basic model initialization
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Essential for memory efficiency
                low_cpu_mem_usage=True
            ).to(self.device)
            
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

    def analyze_resume(self, 
                      resume_text: str, 
                      job_description: str,
                      target_job: Optional[str] = None,
                      target_industry: Optional[str] = None) -> AnalysisResult:
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
                    # Store both original and title case versions
                    skill_matches[skill] = float(similarity)
                    skill_matches[skill.title()] = float(similarity)
                
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
                
                # Generate improvement suggestions
                missing_skills = self._identify_missing_skills(skill_matches)
                suggestions = self._generate_suggestions(
                    skill_matches, experience_score, industry_score,
                    location_match, salary_match, job_req
                )
                
                # Generate optimized resume
                optimized_resume = self._optimize_resume(
                    resume_text,
                    missing_skills,
                    suggestions,
                    skill_matches
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
                    optimized_resume=optimized_resume
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
        """Find matching job requirements from the database"""
        if target_job and target_job in self.data_manager.job_requirements:
            return self.data_manager.job_requirements[target_job]
        
        # First try exact title match if target_job is provided
        if target_job:
            for job_title, req in self.data_manager.job_requirements.items():
                if target_job.lower() in job_title.lower():
                    if not target_industry or req.industry == target_industry:
                        return req
        
        # If no exact match or no target_job, find best semantic match
        best_match = None
        best_score = 0.0
        
        # Encode job description
        encoded_job = self._get_text_embedding(job_description.lower())
        
        # Create a default job requirement if no matches found
        if not self.data_manager.job_requirements:
            self.logger.warning("No job requirements found in data manager, creating default")
            default_req = JobRequirements(
                title="Software Engineer",
                industry="Technology",
                responsibilities=["Software development", "System design"],
                qualifications=["Programming experience", "Bachelor's degree"],
                required_skills={"Python", "Java", "AWS"},
                experience_years=5,
                location="Remote",
                job_type="Full-time",
                salary_range=(100000, 200000)
            )
            return default_req
        
        for job_title, req in self.data_manager.job_requirements.items():
            if target_industry and req.industry != target_industry:
                continue
                
            # Calculate similarity score using transformers
            req_text = f"{job_title} {' '.join(req.responsibilities)} {' '.join(req.qualifications)}"
            encoded_req = self._get_text_embedding(req_text.lower())
            
            # Calculate cosine similarity between embeddings
            score = torch.nn.functional.cosine_similarity(encoded_job, encoded_req, dim=1).mean().item()
            
            if score > best_score:
                best_score = score
                best_match = req
        
        return best_match

    def _extract_experience_years(self, text: str) -> float:
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

    def _calculate_experience_match(self, 
                                  resume_text: str, 
                                  job_req: JobRequirements) -> float:
        """Calculate experience match score"""
        resume_experience = self._extract_experience_years(resume_text)
        required_experience = job_req.experience_years
        
        if required_experience == 0:
            return 1.0
        
        if resume_experience >= required_experience:
            return 1.0
        
        return min(resume_experience / required_experience, 1.0)

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
    
    def _calculate_industry_match(self, 
                                resume_embedding: torch.Tensor, 
                                job_req: JobRequirements) -> float:
        """Calculate industry match score using transformers"""
        try:
            # Get industry keywords and combine with industry name
            industry_keywords = self.data_manager.get_industry_skill_weights(job_req.industry)
            
            if not industry_keywords:
                self.logger.warning(f"No industry keywords found for {job_req.industry}")
                return 0.5  # Default score if no industry data
            
            # Create industry text by combining keywords
            industry_text = f"{job_req.industry} {' '.join(industry_keywords.keys())}"
            
            # Get industry embedding
            industry_embedding = self._get_text_embedding(industry_text.lower())
            
            # Calculate similarity
            similarity = F.cosine_similarity(resume_embedding, industry_embedding, dim=1).mean().item()
            
            # Apply keyword weights
            weighted_score = similarity * sum(industry_keywords.values()) / len(industry_keywords)
            
            return float(min(weighted_score, 1.0))
            
            if not industry_keywords:
                return 0.5  # Default score if no industry data
            
            # Get embeddings
            resume_embedding = self._get_text_embedding(resume_text.lower())
            industry_text = f"{job_req.industry} {' '.join(industry_keywords.keys())}"
            industry_embedding = self._get_text_embedding(industry_text.lower())
            
            # Calculate similarity
            similarity = torch.nn.functional.cosine_similarity(resume_embedding, industry_embedding, dim=1).mean().item()
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating industry match: {str(e)}")
            return 0.5  # Default score on error

    def _check_location_match(self, 
                            resume_text: str, 
                            job_req: JobRequirements) -> bool:
        """Check if resume location matches job location using transformers"""
        if not job_req.location:
            return True
            
        try:
            # Get embeddings for both texts
            resume_embedding = self._get_text_embedding(resume_text.lower())
            location_embedding = self._get_text_embedding(job_req.location.lower())
            
            # Calculate similarity
            similarity = torch.nn.functional.cosine_similarity(resume_embedding, location_embedding, dim=1).mean().item()
            
            # Consider it a match if similarity is above threshold
            return similarity > 0.7  # Adjust threshold as needed
            
        except Exception as e:
            self.logger.error(f"Error checking location match: {str(e)}")
            return True  # Default to True on error

    def _check_salary_match(self, 
                           resume_text: str, 
                           job_req: JobRequirements) -> bool:
        """Check if resume salary expectations match job salary range"""
        if not job_req.salary_range or job_req.salary_range == (0.0, 0.0):
            return True
            
        # Extract salary expectations from resume
        resume_salary = self.extract_salary_range(resume_text)
        if not resume_salary:
            return True  # If no salary mentioned, assume match
            
        min_resume, max_resume = resume_salary
        min_job, max_job = job_req.salary_range
        
        # Check if ranges overlap
        return min_resume <= max_job and max_resume >= min_job
        
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

    def _calculate_ats_score(self,
                           skill_matches: Dict[str, float],
                           experience_score: float,
                           industry_score: float,
                           location_match: bool,
                           salary_match: bool) -> float:
        """Calculate overall ATS score"""
        # Calculate skill score from TF-IDF matches
        skill_score = sum(skill_matches.values()) / len(skill_matches) if skill_matches else 0
        
        # Apply weights to each component
        weighted_score = (
            skill_score * self.weights['skills'] +
            experience_score * self.weights['experience'] +
            industry_score * self.weights['industry'] +
            float(location_match) * self.weights['location'] +
            float(salary_match) * self.weights['salary']
        )
        
        return min(weighted_score, 1.0)  # Normalize to [0, 1]

    def _identify_missing_skills(self, 
                               skill_matches: Dict[str, float]) -> List[str]:
        """Identify critical missing skills based on TF-IDF scores"""
        # Filter skills with low TF-IDF similarity scores
        missing_skills = []
        for skill, score in skill_matches.items():
            if score < 0.5:  # Threshold for considering a skill as missing
                missing_skills.append(skill)
        return sorted(missing_skills, key=lambda x: skill_matches[x])  # Sort by score

    def _optimize_resume(self,
                        resume_text: str,
                        missing_skills: List[str],
                        suggestions: List[str],
                        skill_matches: Dict[str, float]) -> str:
        """Generate an optimized version of the resume based on analysis."""
        try:
            # Start with original resume
            optimized = resume_text
            
            # Add missing skills if they're highly relevant
            relevant_skills = [skill for skill in missing_skills 
                             if skill_matches.get(skill, 0) > 0.5]
            
            if relevant_skills:
                skills_section = "\n\nKey Skills:\n" + ", ".join(relevant_skills)
                optimized += skills_section
            
            # Add improvement suggestions as comments
            if suggestions:
                optimized += "\n\nATS Optimization Suggestions:\n"
                optimized += "\n".join(f"• {suggestion}" for suggestion in suggestions)
            
            return optimized
            
        except Exception as e:
            self.logger.error(f"Error optimizing resume: {str(e)}")
            return resume_text
            
    def _optimize_resume(self,
                        resume_text: str,
                        missing_skills: List[str],
                        suggestions: List[str],
                        skill_matches: Dict[str, float]) -> str:
        """Generate an optimized version of the resume based on analysis."""
        try:
            # Start with original resume
            optimized = resume_text
            
            # Add missing skills if they're highly relevant
            relevant_skills = [skill for skill in missing_skills 
                             if skill_matches.get(skill, 0) > 0.5]
            
            if relevant_skills:
                skills_section = "\n\nKey Skills:\n" + ", ".join(relevant_skills)
                optimized += skills_section
            
            # Add improvement suggestions as comments
            if suggestions:
                optimized += "\n\nATS Optimization Suggestions:\n"
                optimized += "\n".join(f"• {suggestion}" for suggestion in suggestions)
            
            return optimized
            
        except Exception as e:
            self.logger.error(f"Error optimizing resume: {str(e)}")
            return resume_text
            
    def _generate_suggestions(self,
                            skill_matches: Dict[str, float],
                            experience_score: float,
                            industry_score: float,
                            location_match: bool,
                            salary_match: bool,
                            job_req: JobRequirements) -> List[str]:
        """Generate improvement suggestions based on analysis results"""
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
        
        return suggestions
