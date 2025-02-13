import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from pathlib import Path
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import defaultdict
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass
from collections import defaultdict
import json
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.punkt import PunktTokenizer
from nltk.data import find
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

from app.utils.nltk_utils import ensure_nltk_data

# Ensure NLTK data is available
ensure_nltk_data()

@dataclass
class JobRequirements:
    title: str
    industry: str
    responsibilities: List[str]
    qualifications: List[str]
    required_skills: Set[str]
    experience_years: int
    location: str
    job_type: str
    salary_range: Tuple[float, float]
    keyword_density: float = 0.0  # Density of important keywords in the job description
    priority_skills: Set[str] = None  # Skills that are mentioned multiple times or emphasized
    
    def __post_init__(self):
        if self.priority_skills is None:
            self.priority_skills = set()
            
        # Calculate keyword density if not provided
        if self.keyword_density == 0.0:
            text = ' '.join(self.responsibilities + self.qualifications)
            words = word_tokenize(text.lower())
            skill_words = sum(1 for word in words if word in self.required_skills)
            self.keyword_density = (skill_words / len(words)) * 100 if words else 0.0

class DataManager:
    def __init__(self, data_dir: str = "data"):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        
        # Initialize NLTK components with proper error handling
        required_nltk_data = [
            ('tokenizers/punkt', 'punkt'),
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet', 'wordnet'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
        ]
        
        for resource_path, package_name in required_nltk_data:
            try:
                nltk.data.find(resource_path)
                self.logger.debug(f'NLTK resource {package_name} already downloaded')
            except LookupError:
                self.logger.info(f'Downloading NLTK resource {package_name}')
                try:
                    nltk.download(package_name)
                    self.logger.info(f'Successfully downloaded {package_name}')
                except Exception as e:
                    self.logger.error(f'Failed to download {package_name}: {str(e)}')
                    raise
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize TF-IDF vectorizer for skill matching
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),  # Include up to 3-word phrases
            max_features=10000,   # Increase feature space
            analyzer='word',
            min_df=1,            # Minimum document frequency
            max_df=1.0,         # Maximum document frequency
            sublinear_tf=True    # Apply sublinear TF scaling
        )
        
        # Initialize data structures
        self.industry_skills: Dict[str, Dict[str, float]] = {}
        self.skill_patterns: Dict[str, Set[str]] = {}
        self.job_requirements: Dict[str, JobRequirements] = {}
        
        self.initialize_data()

    def initialize_data(self):
        """Load and process sample job data"""
        try:
            # Load sample job data
            sample_data_path = Path(__file__).parent.parent / 'data' / 'sample_jobs.csv'
            if not sample_data_path.exists():
                self.logger.error(f"Sample data file not found at {sample_data_path}")
                raise FileNotFoundError(f"Sample data file not found at {sample_data_path}")
            
            # Load the sample data
            combined_df = pd.read_csv(sample_data_path)
            
            # First build skill patterns
            self._build_skill_patterns()
            
            # Then process job data using the patterns
            self._process_job_data(combined_df)
            
            # Finally calculate industry weights
            self._calculate_industry_weights()
            
            self.logger.info("Data initialization completed successfully")
        except Exception as e:
            self.logger.error(f"Data initialization failed: {str(e)}")
            raise

    def _process_job_data(self, df: pd.DataFrame):
        """Process and merge job description data"""
        try:
            # Required base columns
            required_base_columns = {
                'title', 'industry', 'responsibilities', 'qualifications',
                'location', 'job_type', 'salary_range'
            }
            
            # Optional columns that can be derived
            optional_columns = {
                'required_skills',  # Can be extracted from responsibilities and qualifications
                'experience_years'  # Can be extracted from qualifications
            }
            
            # Column name mapping for standardization
            column_mapping = {
                'Job Title': 'title',
                'Industry': 'industry',
                'Responsibilities': 'responsibilities',
                'Qualifications': 'qualifications',
                'Location': 'location',
                'Job Type': 'job_type',
                'Salary Range': 'salary_range'
            }
            
            # Rename columns if needed
            df = df.rename(columns=column_mapping)
            df_cols = set(df.columns)
            
            # Check for required base columns
            if not required_base_columns <= df_cols:
                missing = required_base_columns - df_cols
                raise ValueError(f"Missing required base columns: {missing}")
            
            for _, row in df.iterrows():
                try:
                    # Process responsibilities and qualifications
                    responsibilities = self._split_text_to_list(row['responsibilities'])
                    qualifications = self._split_text_to_list(row['qualifications'])
                    
                    # Extract experience years from qualifications
                    exp_years = self._extract_experience_years(row['qualifications'])
                    
                    # Extract skills from multiple sources
                    required_skills = set()
                    
                    # Extract from title
                    if not pd.isna(row['title']):
                        title_skills = self._extract_skills(row['title'])
                        required_skills.update(title_skills)
                    
                    # Extract from responsibilities
                    if responsibilities:
                        resp_skills = self._extract_skills(' '.join(responsibilities))
                        required_skills.update(resp_skills)
                    
                    # Extract from qualifications
                    if qualifications:
                        qual_skills = self._extract_skills(' '.join(qualifications))
                        required_skills.update(qual_skills)
                    
                    # Clean and normalize skills
                    cleaned_skills = set()
                    for skill in required_skills:
                        # Convert common variations
                        skill = skill.lower()
                        skill = skill.replace('nodejs', 'node.js')
                        skill = skill.replace('reactjs', 'react.js')
                        skill = skill.replace('vuejs', 'vue.js')
                        skill = skill.replace('machinelearning', 'machine learning')
                        skill = skill.replace('deeplearning', 'deep learning')
                        skill = skill.replace('computervision', 'computer vision')
                        skill = skill.replace('artificialintelligence', 'artificial intelligence')
                        skill = skill.replace('nlp', 'natural language processing')
                        
                        # Remove any extra whitespace
                        skill = ' '.join(skill.split())
                        
                        if len(skill) > 2:  # Ignore very short skills
                            cleaned_skills.add(skill)
                    
                    # Parse salary range
                    salary_range = self._parse_salary_range(row['salary_range'])
                    
                    # Create JobRequirements object
                    job_req = JobRequirements(
                        title=row['title'],
                        industry=row['industry'],
                        responsibilities=responsibilities,
                        qualifications=qualifications,
                        required_skills=required_skills,
                        experience_years=exp_years,
                        location=row['location'],
                        job_type=row['job_type'],
                        salary_range=salary_range
                    )
                    
                    self.job_requirements[job_req.title] = job_req
                    
                except Exception as e:
                    self.logger.warning(f"Error processing row: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error processing job data: {str(e)}")
            raise

    def _extract_experience_years(self, text: str) -> int:
        """Extract years of experience from text using regex patterns"""
        if not text:
            return 0
        
        # Look for patterns like "X+ years" or "X years"
        matches = re.findall(r'(\d+)\+?\s*years?', text.lower())
        
        # Look for work history dates
        work_periods = re.findall(r'(\d{4})\s*-\s*(present|current|\d{4})', text.lower())
        total_years = 0
        
        # Calculate from explicit mentions
        if matches:
            total_years = max(int(year) for year in matches)
        
        # Calculate from work history
        for start, end in work_periods:
            end_year = datetime.now().year if end in ['present', 'current'] else int(end)
            years = end_year - int(start)
            total_years = max(total_years, years)
        
        return total_years

    def _parse_salary_range(self, salary_text: str) -> Tuple[float, float]:
        """Parse salary range from text using enhanced regex patterns"""
        if not salary_text:
            return (0.0, 0.0)
        
        # Handle various salary formats
        patterns = [
            (r'\$(\d+)k\s*-\s*\$(\d+)k', 1000),                    # $70k - $100k
            (r'\$(\d+),000\s*-\s*\$(\d+),000', 1000),             # $70,000 - $100,000
            (r'(\d+)k\s*-\s*(\d+)k', 1000),                       # 70k - 100k
            (r'(\d+),000\s*-\s*(\d+),000', 1000),                 # 70,000 - 100,000
            (r'salary\s*range\s*:\s*\$?(\d+)k?\s*-\s*\$?(\d+)k?', None)  # Salary Range: 70k - 100k
        ]
        
        for pattern, multiplier in patterns:
            matches = re.findall(pattern, salary_text, re.IGNORECASE)
            if matches:
                min_val, max_val = matches[0]
                # If multiplier is None, determine from the text
                if multiplier is None:
                    multiplier = 1000 if 'k' in salary_text.lower() else 1
                min_salary = float(min_val.replace(',', '')) * multiplier
                max_salary = float(max_val.replace(',', '')) * multiplier
                return (min_salary, max_salary)
        
        return (0.0, 0.0)

    def _split_text_to_list(self, text: str) -> List[str]:
        """Split text into list using multiple delimiters"""
        if not text:
            return []
        
        text = str(text)  # Convert to string in case input is None
        items = set()  # Use set to avoid duplicates
        
        # Split by semicolons first
        parts = text.split(';')
        for part in parts:
            # For each part, try sentence tokenization
            try:
                sentences = sent_tokenize(part)
                for sent in sentences:
                    clean = sent.strip().rstrip('.')
                    if clean:
                        items.add(clean)
            except Exception as e:
                # Fallback to period splitting if tokenization fails
                period_parts = part.split('.')
                for p in period_parts:
                    clean = p.strip().rstrip('.')
                    if clean:
                        items.add(clean)
        
        # If we got items, return them as a list
        if items:
            return list(items)
        
        # If no items found but we have text, return it as a single item
        clean = text.strip().rstrip('.')
        return [clean] if clean else []

    def _extract_skills(self, text: str) -> Set[str]:
        """Extract skills from text using NLTK and TF-IDF"""
        if not text:
            return set()

        try:
            # Initialize skills set
            skills = set()
            text_lower = text.lower()

            # First pass: Extract compound technical terms
            compound_patterns = [
                # Programming Languages and Frameworks
                r'\b(python|javascript|typescript|java|c\+\+|ruby|go|rust|swift|kotlin)\b',
                r'\b(node\.js|react\.js|vue\.js|angular\.js|next\.js|express\.js)\b',
                r'\b(django|flask|fastapi|spring|rails)\b',
                
                # AI/ML/Data Science
                r'\b(machine learning|deep learning|artificial intelligence|neural network)\b',
                r'\b(natural language processing|computer vision|data science)\b',
                r'\b(tensorflow|pytorch|scikit[- ]learn|keras|pandas|numpy)\b',
                
                # Databases
                r'\b(sql|mysql|postgresql|mongodb|redis|elasticsearch)\b',
                r'\b(oracle|cassandra|dynamodb|neo4j)\b',
                
                # Cloud and DevOps
                r'\b(aws|azure|gcp|docker|kubernetes|terraform)\b',
                r'\b(jenkins|circleci|travis|gitlab|github actions)\b',
                
                # Web Technologies
                r'\b(html5|css3|sass|less|webpack|babel)\b',
                r'\b(rest api|graphql|websocket|oauth|jwt)\b',
                
                # Software Engineering
                r'\b(git|agile|scrum|ci/cd|tdd|devops)\b',
                r'\b(microservices|serverless|system design)\b'
            ]

            for pattern in compound_patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    skills.add(match.group(0))

            # Second pass: Extract from skill patterns
            for skill, patterns in self.skill_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in text_lower:
                        skills.add(skill.lower())
                        break

            # Third pass: Extract technical terms using NLTK
            text_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', text_lower)
            tokens = word_tokenize(text_clean)
            if tokens:
                pos_tags = pos_tag(tokens)
                current_term = []
                
                for word, pos in pos_tags:
                    if pos in ['NN', 'NNS', 'NNP', 'NNPS'] and word not in self.stop_words:
                        current_term.append(word)
                    else:
                        if current_term:
                            term = ' '.join(current_term)
                            if len(term) > 2:  # Ignore very short terms
                                skills.add(term)
                            current_term = []
                
                if current_term:  # Add last term if exists
                    term = ' '.join(current_term)
                    if len(term) > 2:
                        skills.add(term)

            return skills

            # Extract potential skills (noun phrases and technical terms)
            try:
                # Use NLTK's default English POS tagger
                tagged = nltk.pos_tag(tokens)
            except LookupError:
                try:
                    nltk.download('averaged_perceptron_tagger')
                    tagged = nltk.pos_tag(tokens)
                except Exception as e:
                    self.logger.error(f"Failed to download tagger: {str(e)}")
                    # Fall back to using tokens as skills
                    skills.update(tokens)
                    return skills
            except Exception as e:
                self.logger.warning(f"Error with POS tagger: {str(e)}")
                # Fall back to using tokens as skills
                skills.update(tokens)
                return skills
            
            # Extract noun phrases
            i = 0
            while i < len(tagged):
                if tagged[i][1].startswith('NN'):  # Noun
                    phrase = [tagged[i][0]]
                    j = i + 1
                    # Look for adjacent nouns or adjectives
                    while j < len(tagged) and tagged[j][1].startswith(('NN', 'JJ')):
                        phrase.append(tagged[j][0])
                        j += 1
                    if len(phrase) > 0:
                        skills.add(' '.join(phrase))
                    i = j
                else:
                    i += 1

            # Add single technical terms
            technical_patterns = [
                r'\b[a-z]+script\b',  # JavaScript, TypeScript
                r'\b[a-z]+\.?js\b',   # Node.js, React.js
                r'\b[a-z]+\+\+\b',    # C++
                r'\b[a-z]+#\b',       # C#
                r'\b[a-z]+sql\b',     # MySQL, PostgreSQL
                r'\b[a-z]+db\b',      # MongoDB
                r'\b[a-z]+ml\b',      # HTML, XML
                r'\b[a-z]+api\b',     # REST API, GraphQL API
            ]
            
            for pattern in technical_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    skills.add(match.group(0).lower())
            
            return skills

        except Exception as e:
            self.logger.error(f"Error extracting skills: {str(e)}")
            return set()

    def _build_skill_patterns(self):
        """Build patterns for skill recognition with default technical skills"""
        try:
            # Define default technical skills and their variations
            tech_variations = {
                'Python': ['python', 'py', 'python3'],
                'Java': ['java', 'java programming', 'jdk'],
                'AWS': ['aws', 'amazon web services', 'aws cloud'],
                'SQL': ['sql', 'mysql', 'postgresql', 'database'],
                'Machine Learning': ['machine learning', 'ml', 'deep learning', 'ai'],
                'Statistics': ['statistics', 'statistical analysis', 'data analysis'],
                'Microservices': ['microservices', 'micro-services', 'service oriented'],
                'Project Management': ['project management', 'project lead', 'team lead'],
                'Technical Leadership': ['technical lead', 'tech lead', 'team leadership'],
                'Product Strategy': ['product strategy', 'product development', 'product management']
            }
            
            # Convert variations to pattern sets
            self.skill_patterns = {}
            for skill, variations in tech_variations.items():
                pattern_set = set(variations)
                pattern_set.add(skill.lower())
                self.skill_patterns[skill] = pattern_set
            
            self.logger.info(f"Built patterns for {len(self.skill_patterns)} skills")
            
        except Exception as e:
            self.logger.error(f"Error building skill patterns: {str(e)}")
            raise
            # Initialize patterns dictionary
            self.skill_patterns = {}
            
            # Collect all unique skills
            all_skills = set()
            for job in self.job_requirements.values():
                if job.required_skills:  # Check for None
                    all_skills.update(job.required_skills)
            
            self.logger.info(f"Found {len(all_skills)} unique skills")
            
            # Build variations for each skill
            for skill in all_skills:
                try:
                    variations = {skill.lower()}
                    
                    # Add basic variations
                    variations.add(skill.title())
                    variations.add(skill.upper())
                    
                    # Handle special characters and spacing
                    clean_skill = skill.lower()
                    variations.update([
                        clean_skill.replace(' ', '-'),
                        clean_skill.replace('-', ' '),
                        clean_skill.replace('.', ' '),
                        clean_skill.replace(' ', ''),
                        clean_skill.replace('_', ' '),
                        clean_skill.replace('/', ' '),
                    ])
                    
                    # Add tech variations if present
                    skill_key = next((k for k in tech_variations if k.lower() in clean_skill), None)
                    if skill_key:
                        variations.update(tech_variations[skill_key])
                    
                    # Generate acronym for multi-word skills
                    words = skill.split()
                    if len(words) > 1:
                        # Standard acronym
                        acronym = ''.join(word[0] for word in words)
                        variations.update([acronym.lower(), acronym.upper()])
                        
                        # Partial matches
                        for i in range(len(words)):
                            partial = ' '.join(words[i:])
                            variations.update([partial.lower(), partial.title()])
                    
                    # Lemmatize variations
                    lemmatized = set()
                    for var in variations:
                        try:
                            tokens = word_tokenize(str(var))
                            lemma = ' '.join(self.lemmatizer.lemmatize(token) for token in tokens)
                            if lemma:
                                lemmatized.add(lemma)
                        except Exception as e:
                            self.logger.debug(f"Lemmatization failed for {var}: {str(e)}")
                            continue
                    
                    variations.update(lemmatized)
                    
                    # Remove empty strings and None values
                    variations = {v for v in variations if v and v.strip()}
                    
                    if variations:  # Only add if we have valid variations
                        self.skill_patterns[skill] = variations
                        
                except Exception as e:
                    self.logger.warning(f"Error processing skill {skill}: {str(e)}")
                    continue
            
            self.logger.info(f"Built patterns for {len(self.skill_patterns)} skills")
            
        except Exception as e:
            self.logger.error(f"Error building skill patterns: {str(e)}")
            # Initialize with empty patterns rather than failing
            self.skill_patterns = {}

    def _calculate_industry_weights(self):
        """Calculate skill importance weights by industry using TF-IDF"""
        try:
            # Prepare industry documents
            industry_docs = defaultdict(list)
            for job_req in self.job_requirements.values():
                if not job_req.required_skills:
                    continue
                    
                industry = job_req.industry
                # Create a document with all skills and their variations
                skills_doc = []
                for skill in job_req.required_skills:
                    skills_doc.append(skill)
                    # Add variations from skill patterns
                    if skill in self.skill_patterns:
                        skills_doc.extend(self.skill_patterns[skill])
                
                if skills_doc:  # Only add if we have skills
                    doc = ' '.join(skills_doc)
                    industry_docs[industry].append(doc)
            
            # Calculate TF-IDF for each industry
            for industry, docs in industry_docs.items():
                if not docs:  # Skip if no documents
                    continue
                    
                # Combine all documents for this industry
                combined_doc = ' '.join(docs)
                if not combined_doc.strip():  # Skip if empty after stripping
                    continue
                
                try:
                    # Fit TF-IDF
                    tfidf_matrix = self.vectorizer.fit_transform([combined_doc])
                    feature_names = self.vectorizer.get_feature_names_out()
                    
                    # Calculate weights
                    weights = {}
                    for idx, score in enumerate(tfidf_matrix.toarray()[0]):
                        skill = feature_names[idx]
                        weights[skill] = float(score)
                    
                    # Normalize weights
                    if weights:
                        max_weight = max(weights.values())
                        self.industry_skills[industry] = {
                            skill: weight / max_weight
                            for skill, weight in weights.items()
                        }
                    else:
                        self.logger.warning(f"No weights calculated for industry: {industry}")
                except ValueError as e:
                    self.logger.error(f"Error calculating TF-IDF for industry {industry}: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error calculating industry weights: {str(e)}")
            # Don't raise, allow initialization to continue with default weights

    def get_industry_skill_weights(self, industry: str) -> Dict[str, float]:
        """Get skill importance weights for an industry"""
        return self.industry_skills.get(industry, {})

    def match_skills(self, resume_text: str, job_req: JobRequirements) -> Dict[str, float]:
        """Match resume skills against job requirements using TF-IDF similarity"""
        if not resume_text or not job_req.required_skills:
            return {}

        # Extract skills from resume
        resume_skills = self._extract_skills(resume_text)
        if not resume_skills:
            return {}

        # Get industry weights if available
        industry_weights = self.get_industry_skill_weights(job_req.industry)

        # Calculate match scores
        matches = {}
        for skill in job_req.required_skills:
            # Check for exact match
            if skill.lower() in {s.lower() for s in resume_skills}:
                base_score = 1.0
            else:
                # Check for partial matches using skill patterns
                skill_variations = self.skill_patterns.get(skill, {skill.lower()})
                resume_lower = {s.lower() for s in resume_skills}
                if any(var in resume_lower for var in skill_variations):
                    base_score = 0.9  # High score for pattern match
                else:
                    # Check for word-level matches
                    skill_words = set(skill.lower().split())
                    resume_words = set(' '.join(resume_skills).lower().split())
                    word_matches = len(skill_words & resume_words)
                    if word_matches > 0:
                        base_score = 0.7 * (word_matches / len(skill_words))
                    else:
                        continue

            # Apply industry weight if available
            weight = industry_weights.get(skill.lower(), 1.0) if industry_weights else 1.0
            final_score = min(base_score * weight * 1.2, 1.0)  # Boost score and cap at 1.0
            matches[skill] = final_score

        return matches
        
        # Calculate TF-IDF similarity
        texts = [resume_processed, job_processed]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Calculate similarity scores for each skill
        matches = {}
        industry_weights = self.get_industry_skill_weights(job_req.industry)
        
        for skill in job_skills:
            # Check for exact matches first
            if skill.lower() in resume_skills:
                matches[skill] = industry_weights.get(skill, 0.8)
                continue
            
            # Check variations
            skill_variations = self.skill_patterns.get(skill, {skill})
            matched = False
            for variation in skill_variations:
                if variation.lower() in resume_skills:
                    matched = True
                    break
            
            if matched:
                matches[skill] = industry_weights.get(skill, 0.6)
                continue
            
            # If no direct match, use TF-IDF similarity
            try:
                skill_idx = feature_names.searchsorted(skill.lower())
                if skill_idx < len(feature_names) and feature_names[skill_idx] == skill.lower():
                    skill_score = tfidf_matrix[0, skill_idx] * industry_weights.get(skill, 0.5)
                    matches[skill] = float(skill_score)
                else:
                    matches[skill] = 0.0
            except Exception as e:
                self.logger.warning(f"Error calculating similarity for skill {skill}: {str(e)}")
                matches[skill] = 0.0
        
        return matches

    def save_skill_patterns(self, filepath: str):
        """Save skill patterns to JSON file"""
        # Convert sets to lists for JSON serialization
        serializable_patterns = {k: list(v) if isinstance(v, set) else v 
                               for k, v in self.skill_patterns.items()}
        with open(filepath, 'w') as f:
            json.dump(serializable_patterns, f, indent=2)

    def load_skill_patterns(self, filepath: str):
        """Load skill patterns from JSON file"""
        with open(filepath, 'r') as f:
            # Convert lists back to sets
            patterns = json.load(f)
            self.skill_patterns = {k: set(v) if isinstance(v, list) else v 
                                 for k, v in patterns.items()}
