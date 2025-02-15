import logging
import re
from typing import Dict, List, Tuple
from collections import Counter

class RoleSpecificAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cs_keywords = {
            'metrics': [
                'nps', 'csat', 'churn rate', 'retention rate', 'time to value',
                'product adoption', 'customer health score', 'mrr', 'arr',
                'expansion revenue', 'renewal rate', 'quarterly targets',
                'implementation rate', 'download rate'
            ],
            'tools': [
                'salesforce', 'gainsight', 'totango', 'intercom', 'zendesk',
                'freshdesk', 'hubspot', 'jira', 'confluence', 'segment',
                'microsoft office', 'excel', 'powerpoint', 'teams'
            ],
            'skills': [
                'customer onboarding', 'relationship management', 'account management',
                'stakeholder management', 'technical support', 'product training',
                'customer education', 'escalation management', 'renewal management',
                'upsell strategy', 'customer advocacy', 'strategic planning',
                'portfolio management', 'quarterly business reviews'
            ]
        }

    def analyze_role_match(self, resume_text: str, job_description: str) -> Dict:
        # Validate inputs
        if not resume_text:
            msg = "Empty resume text provided for role-specific analysis"
            self.logger.error(msg)
            raise ValueError(msg)
        if not job_description:
            msg = "Empty job description provided for role-specific analysis"
            self.logger.error(msg)
            raise ValueError(msg)
        
        try:
            # Perform role-specific skill analysis
            skills_analysis = self._analyze_skills(resume_text, job_description)
            
            # Perform role-specific tool analysis
            tools_analysis = self._analyze_tools(resume_text, job_description)
            
            # Perform role-specific metrics analysis
            metrics_analysis = self._analyze_metrics(resume_text, job_description)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(skills_analysis, tools_analysis, metrics_analysis)
            
            return {
                'overall_score': overall_score,
                'metrics': metrics_analysis,
                'tools': tools_analysis,
                'skills': skills_analysis
            }
        except Exception as e:
            self.logger.error("Role-specific analysis error: " + str(e), exc_info=True)
            raise RuntimeError("Role-specific analysis failed") from e


    # Helper function stubs for modular analysis

    def _analyze_skills(self, resume_text: str, job_description: str) -> Dict:
        """Analyze role-specific skills. Placeholder implementation."""
        # TODO: Implement actual skill analysis logic
        return {"score": 0}


    def _analyze_tools(self, resume_text: str, job_description: str) -> Dict:
        """Analyze role-specific tools proficiency. Placeholder implementation."""
        # TODO: Implement actual tool analysis logic
        return {"score": 0}


    def _analyze_metrics(self, resume_text: str, job_description: str) -> Dict:
        """Analyze role-specific metrics. Placeholder implementation."""
        # TODO: Implement actual metrics analysis logic
        return {"score": 0}


    def _calculate_overall_score(self, skills: Dict, tools: Dict, metrics: Dict) -> float:
        """Calculate an overall score based on skills, tools, and metrics analysis."""
        try:
            skill_score = skills.get("score", 0)
            tool_score = tools.get("score", 0)
            metric_score = metrics.get("score", 0)
            overall = (skill_score + tool_score + metric_score) / 3.0
            return overall
        except Exception as e:
            self.logger.error("Error calculating overall score: " + str(e), exc_info=True)
            return 0.0

    def _analyze_category(self, resume_text: str, job_description: str, keywords: List[str]) -> Dict:
        """Analyze matches for a specific category of keywords."""
        # Find required keywords from job description
        required = set(kw for kw in keywords if kw in job_description)
        
        # Find matches in resume
        matches = set(kw for kw in required if kw in resume_text)
        
        # Find additional relevant keywords in resume
        extras = set(kw for kw in keywords if kw in resume_text and kw not in required)
        
        # Calculate missing keywords
        missing = required - matches

        return {
            'matches': list(matches),
            'missing': list(missing),
            'extras': list(extras)
        }

    def _generate_recommendations(self, missing_metrics: List[str],
                                missing_tools: List[str],
                                missing_skills: List[str],
                                role_type: str) -> List[str]:
        """Generate role-specific recommendations based on missing elements."""
        recommendations = []
        
        if missing_metrics:
            recommendations.append(
                f"Add experience with key metrics: {', '.join(missing_metrics[:3])}"
            )
        
        if missing_tools:
            recommendations.append(
                f"Highlight proficiency in tools: {', '.join(missing_tools[:3])}"
            )
        
        if missing_skills:
            recommendations.append(
                f"Emphasize these skills: {', '.join(missing_skills[:3])}"
            )

        # Role-specific recommendations
        if role_type == 'cx':
            recommendations.append(
                "Focus on experience with customer journey mapping and voice of customer programs"
            )
        elif role_type == 'cs':
            recommendations.append(
                "Emphasize customer onboarding and relationship management experience"
            )
        else:
            recommendations.append(
                "Demonstrate ability to bridge CX insights with CS operations"
            )

        return recommendations
