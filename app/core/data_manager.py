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
from .matchers import CombinedMatcher
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
    role_type: str = 'cs'  # Default to Customer Success
    
    def __post_init__(self):
        if self.priority_skills is None:
            self.priority_skills = set()
            
        # Calculate keyword density if not provided
        if self.keyword_density == 0.0:
            text = ' '.join(self.responsibilities + self.qualifications)
            words = word_tokenize(text.lower())
            skill_words = sum(1 for word in words if word in self.required_skills)
            filtered_words = [w for w in words if w.lower() not in self.stop_words]
            self.keyword_density = (skill_words / len(filtered_words)) * 100 if filtered_words else 0.0

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
        
        # Initialize the combined matcher
        self.matcher = CombinedMatcher(
            industry_skills=self.industry_skills,
            cache_dir=self.data_dir / 'cache',
            data_dir=self.data_dir
        )
        
        # Initialize data structures with comprehensive CS/CX skills
        self.industry_skills: Dict[str, Dict[str, float]] = {
            'Customer Success': {
                'customer success': 1.0,
                'account management': 0.95,
                'customer retention': 0.95,
                'client relationship': 0.9,
                'customer experience': 0.9,
                'customer journey': 0.85,
                'salesforce': 0.85,
                'customer satisfaction': 0.85,
                'gainsight': 0.8,
                'zendesk': 0.8,
                'nps': 0.8,
                'csat': 0.8,
                'customer support': 0.75,
                'data analysis': 0.75,
                'project management': 0.7,
                'technical support': 0.7,
                'implementation': 0.7,
                'training': 0.65,
                'documentation': 0.6
            },
            'SaaS': {
                'saas': 1.0,
                'customer success': 0.95,
                'product adoption': 0.9,
                'user onboarding': 0.85,
                'subscription management': 0.8,
                'revenue retention': 0.8,
                'churn reduction': 0.75,
                'customer lifecycle': 0.7
            }
        }
        self.skill_patterns: Dict[str, Set[str]] = {}
        self.job_requirements: Dict[str, JobRequirements] = {}
        
        self.initialize_data()

    def initialize_data(self):
        """Load and process CS/CX job data"""
        try:
            # Create default CS/CX job data
            default_data = {
                'title': ['Customer Success Manager', 'Customer Experience Lead', 
                         'Technical Customer Success Manager', 'Implementation Specialist',
                         'Customer Support Lead', 'Customer Success Operations'],
                'industry': ['SaaS', 'Technology', 'Technology', 'SaaS', 'Technology', 'Technology'],
                'responsibilities': [
                    'Drive customer satisfaction; Manage client relationships; Ensure product adoption',
                    'Develop CX strategy; Analyze customer feedback; Implement improvement initiatives',
                    'Provide technical support; Drive customer enablement; Manage integrations',
                    'Manage implementations; Configure solutions; Train customers',
                    'Lead support team; Resolve escalations; Improve support processes',
                    'Optimize CS operations; Analyze metrics; Drive process improvements'
                ],
                'qualifications': [
                    'Customer success experience; Communication skills; CRM expertise',
                    'Customer experience platforms; Data analysis; Project management',
                    'Technical background; Customer success; API knowledge',
                    'Project management; Technical aptitude; Training experience',
                    'Customer support experience; Leadership skills; Ticketing systems',
                    'Operations experience; Data analysis; CS tools and platforms'
                ],
                'location': ['San Francisco', 'New York', 'Remote', 'Austin', 'Remote', 'Boston'],
                'salary_range': ['$80,000 - $120,000', '$90,000 - $130,000', '$100,000 - $150,000',
                               '$75,000 - $110,000', '$85,000 - $125,000', '$90,000 - $130,000'],
                'job_type': ['Full-time', 'Full-time', 'Full-time', 'Full-time', 'Full-time', 'Full-time']
            }
            
            # Create DataFrame
            df = pd.DataFrame(default_data)
            
            # First build skill patterns
            self._build_skill_patterns()
            
            # Then process job data using the patterns
            self._process_job_data(df)
            
            # Finally calculate industry weights
            self._calculate_industry_weights()
            
            self.logger.info("CS/CX data initialization completed successfully")
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
                'experience_years',  # Can be extracted from qualifications
                'priority_skills',  # Can be determined from emphasis and repetition
                'keyword_density'  # Can be calculated from text
            }
            
            # Validate required columns
            missing_columns = required_base_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Process each job posting
            for idx, row in df.iterrows():
                try:
                    # Extract text for skill analysis
                    text = ' '.join([row['responsibilities'], row['qualifications']])
                    
                    # Extract skills
                    skills = self._extract_skills(text)
                    
                    # Parse salary range
                    salary_min, salary_max = self._parse_salary_range(row['salary_range'])
                    
                    # Create job requirements object
                    job_req = JobRequirements(
                        title=row['title'],
                        industry=row['industry'],
                        responsibilities=row['responsibilities'].split(';'),
                        qualifications=row['qualifications'].split(';'),
                        required_skills=skills,
                        experience_years=self._extract_experience_years(row['qualifications']),
                        location=row['location'],
                        job_type=row['job_type'],
                        salary_range=(salary_min, salary_max)
                    )
                    
                    # Store job requirements
                    self.job_requirements[str(idx)] = job_req
                    
                except Exception as e:
                    self.logger.error(f"Error processing job at index {idx}: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error processing job data: {str(e)}")
            raise
            
    def _get_keyword_weight(self, keyword: str) -> float:
        """Get weight for a keyword based on its importance in CS/CX context"""
        keyword = keyword.lower()
        
        # High importance CS/CX terms (weight 2.0)
        if keyword in {'cs', 'cx', 'customer success', 'customer experience',
                      'retention', 'churn', 'nps', 'csat', 'ces'}:
            return 2.0
            
        # Medium importance terms (weight 1.5)
        if keyword in {'salesforce', 'gainsight', 'zendesk', 'qbr', 'onboarding',
                      'implementation', 'account management'}:
            return 1.5
            
        # Standard terms (weight 1.0)
        return 1.0

    def _get_skill_weight(self, skill: str) -> float:
        """Get weight for a skill based on its importance in CS/CX roles with dynamic weighting"""
        skill = skill.lower()
        base_weight = 1.0
        
        # Get industry-specific weight
        industry_data = self.industry_skills.get(self.industry, {})
        if industry_data:
            total_skills = sum(industry_data.values())
            industry_weight = industry_data.get(skill, 1) / total_skills if total_skills > 0 else 0.5
            base_weight = max(base_weight, industry_weight * 3)  # Scale up to 3.0 max
        
        # Apply multipliers based on skill categories
        multipliers = 1.0
        
        # Critical CS/CX skills
        if any(term in skill for term in [
            'customer success', 'customer experience', 'account management',
            'retention', 'churn prevention', 'client success'
        ]):
            multipliers *= 2.0
        
        # Important CS/CX tools and metrics
        if any(term in skill for term in [
            'salesforce', 'gainsight', 'nps', 'csat', 'qbr',
            'revenue growth', 'customer satisfaction', 'customer engagement'
        ]):
            multipliers *= 1.75
        
        # Technical skills with CS/CX focus
        if any(term in skill for term in [
            'sql', 'analytics', 'reporting', 'bi tools', 'data analysis',
            'customer data', 'metrics tracking', 'kpi monitoring'
        ]):
            multipliers *= 1.5
        
        # Enhanced soft skills for CS/CX
        if any(term in skill for term in [
            'communication', 'presentation', 'leadership', 'stakeholder',
            'relationship building', 'conflict resolution', 'customer advocacy'
        ]):
            multipliers *= 1.25
        
        # Calculate final weight with bounds
        final_weight = min(3.0, max(0.5, base_weight * multipliers))
        
        self.logger.debug(f'Skill weight for {skill}: {final_weight:.2f} '
                         f'(base: {base_weight:.2f}, multipliers: {multipliers:.2f})')
        
        return final_weight

    def _are_skills_similar(self, skill1: str, skill2: str) -> bool:
        """Check if two skills are semantically similar in CS/CX context"""
        # Direct match
        if skill1 == skill2:
            return True
            
        # Common variations
        variations = {
            'customer success': {'cs', 'customer success management', 'csm'},
            'customer experience': {'cx', 'customer experience management'},
            'account management': {'am', 'client management', 'portfolio management'},
            'business intelligence': {'bi', 'analytics', 'reporting'},
            'stakeholder management': {'relationship management', 'client relationship'},
        }
        
        # Check if skills are variations of each other
        for base_skill, variants in variations.items():
            if skill1 in variants and skill2 in variants:
                return True
            
        # Check for partial matches with minimum length
        if len(skill1) > 3 and len(skill2) > 3:
            if skill1 in skill2 or skill2 in skill1:
                return True
                
        return False

    def calculate_match_score(self, job_skills: Set[str], resume_skills: Set[str]) -> float:
        """Calculate enhanced match score with CS/CX specific weighting"""
        if not job_skills:
            return 0.0
            
        try:
            total_weight = 0
            matched_weight = 0
            
            for skill in job_skills:
                # Get skill importance weight
                weight = self._get_skill_weight(skill)
                total_weight += weight
                
                # Check for direct match
                if skill in resume_skills:
                    matched_weight += weight
                    continue
                    
                # Check for semantic matches
                normalized_skill = self._normalize_text(skill)
                for resume_skill in resume_skills:
                    if self._are_skills_similar(normalized_skill, self._normalize_text(resume_skill)):
                        matched_weight += weight * 0.8  # 80% weight for semantic matches
                        break
            
            if total_weight == 0:
                return 0.0
                
            score = (matched_weight / total_weight) * 100
            return min(100.0, max(0.0, score))  # Ensure score is between 0 and 100
            
        except Exception as e:
            self.logger.error(f"Error calculating match score: {str(e)}")
            return 0.0

    def _calculate_keyword_density(self, text: str, keywords: Set[str]) -> float:
        """Calculate keyword density in text with improved normalization"""
        if not text or not keywords:
            return 0.0
        
        try:
            # Normalize text
            text_clean = self._normalize_text(text)
            words = text_clean.split()
            if not words:
                return 0.0
            
            # Normalize keywords
            normalized_keywords = {self._normalize_text(k) for k in keywords}
            
            # Count keyword occurrences with weights
            keyword_count = 0
            for word in words:
                if word in normalized_keywords:
                    # Apply CS/CX specific weighting
                    weight = self._get_keyword_weight(word)
                    keyword_count += weight
            
            # Calculate density with bounds
            density = (keyword_count / len(words)) * 100
            
            # Ensure density falls within desired range (2-4.5%)
            return max(2.0, min(4.5, density))
            
        except Exception as e:
            self.logger.error(f"Error calculating keyword density: {str(e)}")
            return 2.0  # Return minimum acceptable density on error

    def _normalize_text(self, text: str) -> str:
        """Enhanced text normalization with CS/CX specific handling"""
        if not text:
            return ""
            
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Handle common CS/CX abbreviations and variations
            replacements = {
                'customer success': 'cs',
                'customer experience': 'cx',
                'customer support': 'support',
                'account management': 'am',
                'quarterly business review': 'qbr',
                'net promoter score': 'nps',
                'customer satisfaction': 'csat',
                'customer effort score': 'ces',
                'annual recurring revenue': 'arr',
                'monthly recurring revenue': 'mrr'
            }
            
            for full, abbr in replacements.items():
                text = text.replace(full, abbr)
            
            # Remove punctuation while preserving important characters
            text = re.sub(r'[^\w\s\-\.]', ' ', text)
            
            # Normalize spaces
            text = re.sub(r'\s+', ' ', text)
            
            # Remove common stop words while preserving important terms
            words = text.split()
            important_terms = {'cs', 'cx', 'b2b', 'b2c', 'roi', 'kpi', 'api'}
            words = [w for w in words if w in important_terms or len(w) > 2]
            
            return ' '.join(words).strip()
            
        except Exception as e:
            self.logger.error(f"Error normalizing text: {str(e)}")
            return text.lower().strip()
            
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

    def _get_cs_cx_patterns(self) -> List[Tuple[str, float]]:
        """Get weighted patterns for CS/CX skill matching"""
        return [
            # Core CS/CX Skills (Weight: 2.0)
            (r'\b(customer[- ]success|client[- ]success|cs[- ]management|csm|customer[- ]success[- ]manager)\b', 2.0),
            (r'\b(customer[- ]experience|user[- ]experience|cx|ux|customer[- ]experience[- ]manager)\b', 2.0),
            (r'\b(customer[- ]support|technical[- ]support|client[- ]support|support[- ]management)\b', 2.0),
            (r'\b(account[- ]management|client[- ]management|portfolio[- ]management|strategic[- ]account)\b', 2.0),
            (r'\b(relationship[- ]management|client[- ]relationship|stakeholder[- ]management|partner[- ]management)\b', 2.0),
            
            # CS/CX Tools and Platforms (Weight: 2.0)
            (r'\b(salesforce|sfdc|crm|sales[- ]cloud|service[- ]cloud|salesforce[- ]admin)\b', 2.0),
            (r'\b(gainsight|zendesk|intercom|freshdesk|hubspot|customer[- ]success[- ]platform)\b', 2.0),
            (r'\b(totango|churnzero|planhat|vitally|catalyst|customer[- ]success[- ]software)\b', 2.0),
            (r'\b(jira|confluence|slack|teams|zoom|collaboration[- ]tools)\b', 2.0),
            
            # CS/CX Metrics and KPIs (Weight: 2.0)
            (r'\b(nps|csat|ces|customer[- ]satisfaction|effort[- ]score|satisfaction[- ]metrics)\b', 2.0),
            (r'\b(churn[- ]rate|retention[- ]rate|revenue[- ]retention|customer[- ]retention)\b', 2.0),
            (r'\b(mrr|arr|ltv|cac|roi|qbr|revenue[- ]metrics|business[- ]metrics)\b', 2.0),
            (r'\b(quarterly[- ]business[- ]review|business[- ]review|qbr|executive[- ]review)\b', 2.0),
            
            # CS/CX Processes (Weight: 2.0)
            (r'\b(onboarding|implementation|product[- ]adoption|customer[- ]onboarding)\b', 2.0),
            (r'\b(customer[- ]journey|user[- ]journey|lifecycle|journey[- ]mapping)\b', 2.0),
            (r'\b(escalation[- ]management|issue[- ]resolution|problem[- ]resolution)\b', 2.0),
            (r'\b(training|enablement|coaching|mentoring|customer[- ]education)\b', 2.0),
            (r'\b(upsell|cross[- ]sell|expansion|growth|revenue[- ]growth)\b', 2.0),
            
            # Data and Analytics (Weight: 1.5)
            (r'\b(data[- ]analysis|analytics|bi|reporting|data[- ]visualization)\b', 1.5),
            (r'\b(dashboards|metrics|kpis|reporting|performance[- ]metrics)\b', 1.5),
            (r'\b(sql|excel|tableau|looker|power[- ]bi|data[- ]tools)\b', 1.5),
            
            # Soft Skills (Weight: 1.5)
            (r'\b(communication|presentation|negotiation|interpersonal)\b', 1.5),
            (r'\b(leadership|management|team[- ]lead|people[- ]management)\b', 1.5),
            (r'\b(problem[- ]solving|critical[- ]thinking|analytical[- ]thinking)\b', 1.5),
            (r'\b(project[- ]management|program[- ]management|delivery[- ]management)\b', 1.5),
            
            # Technical Skills (Weight: 1.5)
            (r'\b(api|integration|automation|workflow|technical[- ]skills)\b', 1.5),
            (r'\b(python|javascript|java|sql|ruby|programming)\b', 1.5),
            (r'\b(aws|azure|cloud|saas|software|cloud[- ]platforms)\b', 1.5),
            (r'\b(data[- ]science|machine[- ]learning|ai|analytics[- ]tools)\b', 1.5)
        ]

    def _extract_skills(self, text: str) -> Set[str]:
        """Extract skills from text using the combined matcher"""
        if not text:
            return set()

        text_clean = self._normalize_text(text)
        return self.matcher.extract_skills(text_clean)s.items():
                try:
                    skill_weight = self._get_skill_weight(skill)
                    if skill_weight >= 1.5:  # Only process high and medium priority skills
                        for pattern in patterns:
                            if pattern.lower() in text_lower:
                                skills.add(skill.lower())
                                break
                except Exception as e:
                    self.logger.debug(f"Skill pattern matching failed for {skill}: {str(e)}")
                    continue

            # Third pass: Extract noun phrases using NLTK with CS/CX focus
            try:
                # Clean text and tokenize
                text_clean = re.sub(r'[^a-zA-Z0-9\s-]', ' ', text_lower)
                tokens = word_tokenize(text_clean)
                
                if tokens:
                    # Perform POS tagging
                    pos_tags = pos_tag(tokens)
                    current_term = []
                    
                    for word, pos in pos_tags:
                        # Focus on nouns and adjectives that might be skills
                        if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ'] and word not in self.stop_words:
                            current_term.append(word)
                        else:
                            if current_term:
                                term = ' '.join(current_term)
                                if len(term) > 2:  # Ignore very short terms
                                    # Only add if it's a CS/CX relevant term
                                    if self._get_skill_weight(term) >= 1.5:
                                        skills.add(term)
                                current_term = []
                    
                    # Add last term if exists
                    if current_term:
                        term = ' '.join(current_term)
                        if len(term) > 2 and self._get_skill_weight(term) >= 1.5:
                            skills.add(term)
            except Exception as e:
                self.logger.warning(f"NLTK processing failed: {str(e)}")

            # Clean and normalize final skill set
            cleaned_skills = set()
            for skill in skills:
                try:
                    # Normalize skill text
                    skill = skill.lower().strip()
                    skill = re.sub(r'\s+', ' ', skill)  # Replace multiple spaces
                    
                    # Skip if too short or contains unwanted characters
                    if len(skill) <= 2 or not re.match(r'^[a-z0-9\s\-\.]+$', skill):
                        continue
                    
                    # Only add skills with sufficient weight
                    if self._get_skill_weight(skill) >= 1.5:
                        cleaned_skills.add(skill)
                except Exception as e:
                    self.logger.debug(f"Skill cleaning failed for {skill}: {str(e)}")
                    continue

            return cleaned_skills

        except Exception as e:
            self.logger.error(f"Error extracting skills: {str(e)}")
            return set()

            # Extract skills using patterns
            for pattern in compound_patterns:
                try:
                    matches = re.finditer(pattern, text_lower)
                    for match in matches:
                        skill = match.group(0).strip()
                        if skill:  # Ensure non-empty skill
                            skills.add(skill)
                except Exception as e:
                    self.logger.debug(f"Pattern matching failed for {pattern}: {str(e)}")
                    continue

            # Second pass: Extract from skill patterns dictionary
            for skill, patterns in self.skill_patterns.items():
                try:
                    for pattern in patterns:
                        if pattern.lower() in text_lower:
                            skills.add(skill.lower())
                            break
                except Exception as e:
                    self.logger.debug(f"Skill pattern matching failed for {skill}: {str(e)}")
                    continue

            # Third pass: Extract noun phrases using NLTK
            try:
                # Clean text and tokenize
                text_clean = re.sub(r'[^a-zA-Z0-9\s-]', ' ', text_lower)
                tokens = word_tokenize(text_clean)
                
                if tokens:
                    # Perform POS tagging
                    pos_tags = pos_tag(tokens)
                    current_term = []
                    
                    for word, pos in pos_tags:
                        if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ'] and word not in self.stop_words:
                            current_term.append(word)
                        else:
                            if current_term:
                                term = ' '.join(current_term)
                                if len(term) > 2:  # Ignore very short terms
                                    skills.add(term)
                                current_term = []
                    
                    # Add last term if exists
                    if current_term:
                        term = ' '.join(current_term)
                        if len(term) > 2:
                            skills.add(term)
            except Exception as e:
                self.logger.warning(f"NLTK processing failed: {str(e)}")

            # Clean and normalize final skill set
            cleaned_skills = set()
            for skill in skills:
                try:
                    # Normalize skill text
                    skill = skill.lower().strip()
                    skill = re.sub(r'\s+', ' ', skill)  # Replace multiple spaces
                    
                    # Skip if too short or contains unwanted characters
                    if len(skill) <= 2 or not re.match(r'^[a-z0-9\s\-\.]+$', skill):
                        continue
                    
                    cleaned_skills.add(skill)
                except Exception as e:
                    self.logger.debug(f"Skill cleaning failed for {skill}: {str(e)}")
                    continue

            return cleaned_skills

        except Exception as e:
            self.logger.error(f"Error extracting skills: {str(e)}")
            return set()

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
        """Build patterns for skill recognition with CS/CX and technical skills"""
        try:
            # Define skills and their variations
            skill_variations = {
                # Technical Skills
                'Python': ['python', 'py', 'python3', 'python programming'],
                'Java': ['java', 'java programming', 'jdk', 'java development'],
                'AWS': ['aws', 'amazon web services', 'aws cloud', 'amazon cloud'],
                'SQL': ['sql', 'mysql', 'postgresql', 'database', 'sql database'],
                'Machine Learning': ['machine learning', 'ml', 'deep learning', 'ai', 'artificial intelligence'],
                'Data Analysis': ['data analysis', 'analytics', 'data analytics', 'business intelligence', 'bi'],
                'API Integration': ['api', 'rest api', 'api integration', 'web services', 'api development'],
                
                # CS/CX Core Skills
                'Customer Success': ['customer success', 'cs', 'csm', 'customer success management', 'client success'],
                'Customer Experience': ['customer experience', 'cx', 'ce', 'user experience', 'ux'],
                'Account Management': ['account management', 'client management', 'portfolio management', 'relationship management'],
                'Customer Support': ['customer support', 'technical support', 'client support', 'customer service'],
                
                # CS/CX Tools
                'Salesforce': ['salesforce', 'crm', 'sfdc', 'sales cloud', 'service cloud'],
                'Gainsight': ['gainsight', 'customer success platform', 'cs platform'],
                'Zendesk': ['zendesk', 'help desk', 'support platform', 'ticket management'],
                'Intercom': ['intercom', 'customer messaging', 'customer engagement platform'],
                
                # CS/CX Metrics
                'NPS': ['nps', 'net promoter score', 'customer satisfaction metric'],
                'CSAT': ['csat', 'customer satisfaction score', 'satisfaction rating'],
                'CES': ['ces', 'customer effort score', 'effort score'],
                'Churn': ['churn', 'churn rate', 'customer churn', 'retention rate'],
                'Revenue Metrics': ['mrr', 'arr', 'monthly recurring revenue', 'annual recurring revenue'],
                'Customer Metrics': ['ltv', 'cac', 'customer lifetime value', 'customer acquisition cost'],
                
                # CS/CX Processes
                'Customer Onboarding': ['onboarding', 'customer onboarding', 'implementation', 'user onboarding'],
                'Product Training': ['product training', 'customer training', 'user training', 'enablement'],
                'Customer Success Ops': ['cs operations', 'success operations', 'cs ops', 'customer success operations'],
                'Business Reviews': ['qbr', 'quarterly business review', 'business review', 'executive review'],
                
                # Soft Skills
                'Communication': ['communication', 'written communication', 'verbal communication', 'presentation skills'],
                'Leadership': ['leadership', 'team leadership', 'people management', 'mentoring'],
                'Problem Solving': ['problem solving', 'critical thinking', 'analytical skills', 'troubleshooting'],
                'Project Management': ['project management', 'program management', 'project coordination']
            }
            
            # Convert variations to pattern sets
            self.skill_patterns = {}
            for skill, variations in skill_variations.items():
                # Create base pattern set
                pattern_set = set(variations)
                pattern_set.add(skill.lower())
                
                # Add common variations
                for variation in list(pattern_set):  # Create a copy to avoid modification during iteration
                    # Add hyphenated and no-space versions
                    if ' ' in variation:
                        pattern_set.add(variation.replace(' ', '-'))
                        pattern_set.add(variation.replace(' ', ''))
                    
                    # Add customer/client variations for CS/CX terms
                    if 'customer' in variation:
                        pattern_set.add(variation.replace('customer', 'client'))
                    if 'client' in variation:
                        pattern_set.add(variation.replace('client', 'customer'))
                
                self.skill_patterns[skill] = pattern_set
            
            self.logger.info(f"Built patterns for {len(self.skill_patterns)} skills")
            
        except Exception as e:
            self.logger.error(f"Error building skill patterns: {str(e)}")
            # Initialize with empty patterns rather than failing
            self.skill_patterns = {}
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
        """Match resume skills against job requirements using the combined matcher"""
        if not resume_text or not job_req.required_skills:
            return {}

        # Extract skills from resume
        resume_skills = self._extract_skills(resume_text)
        if not resume_skills:
            return {}

        try:
            # Use combined matcher for skill matching
            match_scores = self.matcher.match(job_req.required_skills, resume_skills)
            
            # Log detailed matching information
            self.logger.debug(
                f"Skill matching - Resume skills: {resume_skills}, "
                f"Job skills: {job_req.required_skills}, "
                f"Match scores: {match_scores}"
            )
            
            return match_scores
        except Exception as e:
            self.logger.error(f"Error in skill matching: {str(e)}")
            return {}
        
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
