"""
CS/CX-focused validation suite for ATS skill matching
"""
from typing import Dict, Set, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ValidationMetrics:
    """Metrics for skill matching validation"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    skill_coverage: Dict[str, float]
    memory_usage: float
    processing_time: float
    batch_performance: float

class CSValidator:
    """Validator for CS/CX skill matching"""
    
    def __init__(self, data_dir: str = "data"):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        
        # CS/CX-specific validation thresholds
        self.thresholds = {
            'accuracy': 0.90,
            'precision': 0.92,
            'recall': 0.88,
            'f1_score': 0.90,
            'false_positive_rate': 0.05,
            'memory_usage': 500,  # MB
            'processing_time': 100,  # ms
            'batch_performance': 1000  # docs/sec
        }
        
        # Initialize CS/CX skill taxonomies
        self.skill_taxonomies = {
            'core_cs': {
                'customer success',
                'account management',
                'client relationship',
                'customer experience',
                'customer satisfaction'
            },
            'tools': {
                'salesforce',
                'gainsight',
                'zendesk',
                'intercom',
                'hubspot'
            },
            'metrics': {
                'nps',
                'csat',
                'churn rate',
                'mrr',
                'customer health'
            },
            'soft_skills': {
                'communication',
                'leadership',
                'problem solving',
                'presentation',
                'stakeholder management'
            }
        }
        
        # Initialize validation datasets
        self.validation_data = self._load_validation_data()
        
    def _load_validation_data(self) -> Dict[str, List[Tuple[str, Set[str]]]]:
        """Load validation datasets with ground truth skill annotations"""
        validation_data = defaultdict(list)
        
        # Use in-memory test data for validation
        test_jobs = [
            ("We are seeking a Customer Success Manager to join our team. The ideal candidate will have experience with Salesforce, Gainsight, and strong communication skills. Must be able to manage customer relationships, reduce churn, and improve customer satisfaction scores.",
             {"customer success", "salesforce", "gainsight", "communication", "customer satisfaction", "churn reduction", "relationship management"}),
            
            ("Senior Customer Experience Manager needed. Must have proven track record in implementing customer success programs, driving product adoption, and managing stakeholder relationships. Experience with Zendesk and measuring NPS required.",
             {"customer experience", "customer success", "product adoption", "stakeholder management", "zendesk", "nps"}),
             
            ("Customer Success Representative position available. Will be responsible for customer onboarding, maintaining high retention rates, and supporting product implementation. Must be proficient in Hubspot and have excellent presentation skills.",
             {"customer success", "customer onboarding", "customer retention", "implementation", "hubspot", "presentation"})
        ]
        
        test_resumes = [
            ("Experienced Customer Success Manager with 5 years of experience. Proficient in Salesforce, Gainsight, and Intercom. Proven track record of reducing churn by 25% and maintaining 95% customer satisfaction rate.",
             {"customer success", "salesforce", "gainsight", "intercom", "churn reduction", "customer satisfaction"}),
             
            ("Customer Experience professional specializing in SaaS. Expert in Zendesk and measuring customer health metrics. Successfully implemented customer success programs resulting in 40% increase in NPS scores. Strong focus on stakeholder management and product adoption.",
             {"customer experience", "zendesk", "customer health", "customer success", "nps", "stakeholder management", "product adoption"})
        ]
        
        validation_data['jobs'] = test_jobs
        validation_data['resumes'] = test_resumes
        
        self.logger.info(f"Loaded {len(validation_data['jobs'])} job descriptions and "
                        f"{len(validation_data['resumes'])} resumes for validation")
            
        return validation_data
    
    def validate_skill_extraction(self, matcher) -> ValidationMetrics:
        """Validate skill extraction accuracy"""
        results = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'processing_times': [],
            'memory_usage': [],
            'batch_sizes': []
        }
        
        # Test job descriptions and resumes
        for dataset in ['jobs', 'resumes']:
            for text, true_skills in self.validation_data[dataset]:
                start_time = pd.Timestamp.now()
                
                # Normalize true skills
                true_skills = {matcher._normalize_skill(skill) for skill in true_skills}
                
                # Extract and normalize skills
                extracted_skills = matcher.extract_skills(text)
                extracted_skills = {matcher._normalize_skill(skill) for skill in extracted_skills}
                
                processing_time = (pd.Timestamp.now() - start_time).total_seconds() * 1000
                
                # Calculate metrics
                true_positives = len(true_skills & extracted_skills)
                false_positives = len(extracted_skills - true_skills)
                false_negatives = len(true_skills - extracted_skills)
                
                results['true_positives'] += true_positives
                results['false_positives'] += false_positives
                results['false_negatives'] += false_negatives
                results['processing_times'].append(processing_time)
                results['memory_usage'].append(self._get_memory_usage())
        
        # Calculate final metrics
        total_predictions = results['true_positives'] + results['false_positives']
        total_actual = results['true_positives'] + results['false_negatives']
        
        # Handle division by zero
        if total_predictions == 0:
            precision = 1.0  # If no predictions, consider it perfect precision
        else:
            precision = results['true_positives'] / total_predictions
            
        if total_actual == 0:
            recall = 1.0  # If no actual skills, consider it perfect recall
        else:
            recall = results['true_positives'] / total_actual
            
        # Calculate F1 score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        # Calculate accuracy
        total_cases = total_actual + results['false_positives']
        if total_cases == 0:
            accuracy = 1.0
        else:
            accuracy = results['true_positives'] / total_cases
            
        # Calculate false positive rate
        if total_predictions == 0:
            fpr = 0.0
        else:
            fpr = results['false_positives'] / total_predictions
        
        # Calculate skill coverage by category
        skill_coverage = self._calculate_skill_coverage(matcher)
        
        # Calculate batch performance
        total_time = sum(results['processing_times']) / 1000  # Convert to seconds
        total_docs = len(self.validation_data['jobs']) + len(self.validation_data['resumes'])
        batch_performance = total_docs / total_time if total_time > 0 else 0
        
        return ValidationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_positive_rate=fpr,
            skill_coverage=skill_coverage,
            memory_usage=np.mean(results['memory_usage']),
            processing_time=np.mean(results['processing_times']),
            batch_performance=batch_performance
        )
    
    def _calculate_skill_coverage(self, matcher) -> Dict[str, float]:
        """Calculate skill coverage for each CS/CX skill category"""
        coverage = {}
        
        for category, skills in self.skill_taxonomies.items():
            matched_skills = 0
            total_skills = len(skills)
            
            for skill in skills:
                # Create test phrases that include the skill
                test_phrases = [
                    f"Experience with {skill}",
                    f"Proficient in {skill}",
                    f"Skilled in {skill}",
                    f"Knowledge of {skill}",
                    f"Background in {skill}"
                ]
                
                # Test each phrase
                for phrase in test_phrases:
                    extracted = matcher.extract_skills(phrase)
                    if any(extracted):
                        matched_skills += 1
                        break
            
            coverage[category] = matched_skills / total_skills if total_skills > 0 else 0.0
            
        return coverage
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def generate_validation_report(self, metrics: ValidationMetrics) -> str:
        """Generate detailed validation report"""
        report = ["CS/CX Skill Matching Validation Report", "=" * 40, ""]
        
        # Add performance metrics
        report.extend([
            "Performance Metrics:",
            f"- Accuracy: {metrics.accuracy:.2%} (Target: {self.thresholds['accuracy']:.0%})",
            f"- Precision: {metrics.precision:.2%} (Target: {self.thresholds['precision']:.0%})",
            f"- Recall: {metrics.recall:.2%} (Target: {self.thresholds['recall']:.0%})",
            f"- F1 Score: {metrics.f1_score:.2%} (Target: {self.thresholds['f1_score']:.0%})",
            f"- False Positive Rate: {metrics.false_positive_rate:.2%} (Target: {self.thresholds['false_positive_rate']:.0%})",
            ""
        ])
        
        # Add skill coverage
        report.extend(["Skill Coverage by Category:"])
        for category, coverage in metrics.skill_coverage.items():
            report.append(f"- {category}: {coverage:.2%}")
        report.append("")
        
        # Add system metrics
        report.extend([
            "System Metrics:",
            f"- Memory Usage: {metrics.memory_usage:.1f}MB (Target: {self.thresholds['memory_usage']}MB)",
            f"- Processing Time: {metrics.processing_time:.1f}ms (Target: {self.thresholds['processing_time']}ms)",
            f"- Batch Performance: {metrics.batch_performance:.1f} docs/sec (Target: {self.thresholds['batch_performance']} docs/sec)",
            ""
        ])
        
        # Add recommendations
        report.extend(["Recommendations:"])
        if metrics.accuracy < self.thresholds['accuracy']:
            report.append("- Improve overall accuracy through enhanced pattern matching")
        if metrics.false_positive_rate > self.thresholds['false_positive_rate']:
            report.append("- Reduce false positives by tightening matching criteria")
        if metrics.memory_usage > self.thresholds['memory_usage']:
            report.append("- Optimize memory usage through better resource management")
        if metrics.processing_time > self.thresholds['processing_time']:
            report.append("- Improve processing time through caching and optimization")
            
        return "\n".join(report)
