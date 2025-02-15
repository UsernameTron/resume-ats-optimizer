"""
Enhanced metric detection for Customer Success and Customer Experience roles.
Focuses on key CS/CX metrics including satisfaction scores, efficiency metrics,
volume indicators, and impact measurements.
"""
from typing import Dict, List, Tuple, Any, Optional
import re
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MetricMatch:
    value: str
    category: str
    subcategory: str
    confidence: float
    context: str
    timeframe: Optional[str] = None

class CSMetricDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize metric patterns with subcategories
        self.metric_patterns = {
            'satisfaction': {
                'csat': [
                    r'(?:CSAT|satisfaction) (?:score|rating) of (\d+(?:\.\d+)?%?)',
                    r'(\d+(?:\.\d+)?%?) (?:customer satisfaction|CSAT)',
                    r'improved CSAT (?:by|to) (\d+(?:\.\d+)?%?)'
                ],
                'nps': [
                    r'NPS (?:of|at|reached) ([-+]?\d+)',
                    r'increased NPS (?:by|to) ([-+]?\d+)',
                    r'Net Promoter Score of ([-+]?\d+)'
                ],
                'retention': [
                    r'(\d+(?:\.\d+)?%?) retention rate',
                    r'reduced churn (?:by|to) (\d+(?:\.\d+)?%?)',
                    r'customer retention increased to (\d+(?:\.\d+)?%?)'
                ]
            },
            'impact': {
                'adoption': [
                    r'(\d+(?:\.\d+)?%?) adoption rate',
                    r'increased adoption (?:by|to) (\d+(?:\.\d+)?%?)',
                    r'product adoption grew (?:by|to) (\d+(?:\.\d+)?%?)'
                ],
                'engagement': [
                    r'(\d+(?:\.\d+)?[KMB]?) (?:monthly|daily) active users',
                    r'user engagement increased (?:by|to) (\d+(?:\.\d+)?%?)',
                    r'(\d+(?:\.\d+)?%?) user engagement rate'
                ],
                'growth': [
                    r'user base grew (?:to|by) (\d+(?:\.\d+)?[KMB]?)',
                    r'expanded customer base (?:by|to) (\d+(?:\.\d+)?[KMB]?)',
                    r'(\d+(?:\.\d+)?%?) growth in customer base'
                ]
            },
            'efficiency': {
                'resolution': [
                    r'(\d+(?:\.\d+)?%?) first contact resolution',
                    r'average resolution time of (\d+(?:\.\d+)?) (?:minutes|hours)',
                    r'reduced resolution time (?:by|to) (\d+(?:\.\d+)?%?)'
                ],
                'response': [
                    r'(\d+(?:\.\d+)?%?) response rate',
                    r'average response time of (\d+(?:\.\d+)?) (?:minutes|hours)',
                    r'improved response time (?:by|to) (\d+(?:\.\d+)?%?)'
                ],
                'automation': [
                    r'(\d+(?:\.\d+)?%?) automation rate',
                    r'automated (\d+(?:\.\d+)?%?) of processes',
                    r'increased automation (?:by|to) (\d+(?:\.\d+)?%?)'
                ]
            },
            'volume': {
                'tickets': [
                    r'handled (\d+(?:\.\d+)?[KMB]?) tickets',
                    r'processed (\d+(?:\.\d+)?[KMB]?) support requests',
                    r'managed (\d+(?:\.\d+)?[KMB]?) cases'
                ],
                'team': [
                    r'team of (\d+(?:\.\d+)?[KMB]?) (?:agents|representatives)',
                    r'managed (\d+(?:\.\d+)?[KMB]?) team members',
                    r'led (\d+(?:\.\d+)?[KMB]?) (?:person|member) team'
                ],
                'customers': [
                    r'supported (\d+(?:\.\d+)?[KMB]?) customers',
                    r'served (\d+(?:\.\d+)?[KMB]?) clients',
                    r'managed (\d+(?:\.\d+)?[KMB]?) accounts'
                ]
            }
        }
        
        # Value range validation
        self.value_ranges = {
            'satisfaction': {
                'csat': (0, 100),
                'nps': (-100, 100),
                'retention': (0, 100)
            },
            'impact': {
                'adoption': (0, 100),
                'engagement': (0, float('inf')),
                'growth': (0, float('inf'))
            },
            'efficiency': {
                'resolution': (0, 100),
                'response': (0, 100),
                'automation': (0, 100)
            },
            'volume': {
                'tickets': (1, float('inf')),
                'team': (1, 10000),
                'customers': (1, float('inf'))
            }
        }

    def extract_metrics(self, text: str) -> List[MetricMatch]:
        """Extract CS/CX metrics from text with validation and confidence scoring."""
        metrics = []
        
        for category, subcategories in self.metric_patterns.items():
            for subcategory, patterns in subcategories.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        value = match.group(1)
                        if self._validate_metric_value(value, category, subcategory):
                            context = self._get_context(match.start(), text)
                            confidence = self._calculate_confidence(
                                value, category, subcategory, context
                            )
                            timeframe = self._extract_timeframe(context)
                            
                            metrics.append(MetricMatch(
                                value=value,
                                category=category,
                                subcategory=subcategory,
                                confidence=confidence,
                                context=context,
                                timeframe=timeframe
                            ))
        
        return metrics

    def _normalize_value(self, value: str) -> float:
        """Normalize metric values to standard format."""
        try:
            # Handle suffixes
            multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
            value = value.upper().strip()
            
            # Remove plus signs from numbers
            value = value.replace('+', '')
            
            # Handle suffixes
            for suffix, multiplier in multipliers.items():
                if suffix in value:
                    return float(value.replace(suffix, '')) * multiplier
            
            # Handle percentages
            if '%' in value:
                return float(value.rstrip('%'))
            
            return float(value)
        except ValueError as e:
            self.logger.warning(f"Failed to normalize value {value}: {str(e)}")
            return 0.0

    def _validate_metric_value(self, value: str, category: str, subcategory: str) -> bool:
        """Validate if a metric value is within reasonable bounds."""
        try:
            normalized = self._normalize_value(value)
            value_range = self.value_ranges[category][subcategory]
            return value_range[0] <= normalized <= value_range[1]
        except (ValueError, KeyError) as e:
            self.logger.warning(f"Validation failed for {value}: {str(e)}")
            return False

    def _get_context(self, position: int, text: str, window: int = 100) -> str:
        """Get surrounding context for a metric mention."""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end]

    def _calculate_confidence(self, value: str, category: str, 
                            subcategory: str, context: str) -> float:
        """Calculate confidence score for a metric based on multiple factors."""
        confidence = 0.7  # Base confidence
        
        # Boost for professional context
        prof_terms = ['achieved', 'improved', 'increased', 'reduced', 'managed']
        if any(term in context.lower() for term in prof_terms):
            confidence += 0.1
        
        # Boost for specific timeframes
        time_terms = ['monthly', 'quarterly', 'annually', 'year-over-year']
        if any(term in context.lower() for term in time_terms):
            confidence += 0.1
        
        # Boost for CS/CX specific terminology
        cs_terms = ['CSAT', 'NPS', 'customer', 'satisfaction', 'resolution']
        if any(term.upper() in context.upper() for term in cs_terms):
            confidence += 0.1
        
        # Penalize if value seems unrealistic
        try:
            normalized = self._normalize_value(value)
            range_max = self.value_ranges[category][subcategory][1]
            if range_max != float('inf') and normalized > range_max * 0.9:
                confidence -= 0.2
        except (ValueError, KeyError):
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)

    def _extract_timeframe(self, context: str) -> Optional[str]:
        """Extract timeframe information from metric context."""
        timeframe_patterns = [
            r'(?:in|over|during|for) (?:the )?(?:last|past) (\d+ (?:month|year|quarter)s?)',
            r'(?:monthly|quarterly|annual|yearly)',
            r'(?:YoY|MoM|QoQ)',
            r'year[- ](?:over|to)[- ]year',
            r'month[- ](?:over|to)[- ]month'
        ]
        
        for pattern in timeframe_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(0)
        return None
