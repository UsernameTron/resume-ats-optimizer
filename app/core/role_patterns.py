"""Role-specific patterns and utilities for CX and CS analysis."""

from typing import Dict, List, Optional, Set

# Role-specific patterns for Customer Experience (CX) roles
CX_PATTERNS = {
    'keywords': [
        'customer experience',
        'cx',
        'customer satisfaction',
        'csat',
        'customer journey',
        'voice of customer',
        'voc',
        'customer feedback',
        'customer advocacy',
        'customer engagement',
        'customer retention',
        'customer success',
        'customer service',
        'customer support',
        'customer relationship',
        'customer lifecycle',
        'customer-centric',
        'customer-focused'
    ],
    'skills': [
        'empathy',
        'communication',
        'active listening',
        'problem-solving',
        'conflict resolution',
        'customer advocacy',
        'relationship building',
        'stakeholder management',
        'project management',
        'data analysis',
        'customer journey mapping',
        'process improvement',
        'change management',
        'presentation skills',
        'training',
        'mentoring'
    ],
    'tools': [
        'zendesk',
        'salesforce',
        'intercom',
        'freshdesk',
        'gainsight',
        'totango',
        'qualtrics',
        'surveymonkey',
        'hubspot',
        'tableau',
        'power bi',
        'excel',
        'jira',
        'confluence',
        'slack'
    ],
    'metrics': [
        'nps',
        'csat',
        'customer effort score',
        'ces',
        'customer lifetime value',
        'clv',
        'churn rate',
        'retention rate',
        'customer satisfaction score',
        'first contact resolution',
        'fcr',
        'average handle time',
        'aht',
        'customer health score'
    ]
}

# Role-specific patterns for Customer Success (CS) roles
CS_PATTERNS = {
    'keywords': [
        'customer success',
        'cs',
        'account management',
        'client success',
        'client management',
        'portfolio management',
        'customer onboarding',
        'customer adoption',
        'customer growth',
        'revenue retention',
        'upsell',
        'cross-sell',
        'expansion revenue',
        'renewal',
        'product adoption',
        'customer health'
    ],
    'skills': [
        'account management',
        'relationship management',
        'strategic planning',
        'business acumen',
        'sales',
        'negotiation',
        'product expertise',
        'technical support',
        'project management',
        'data analysis',
        'forecasting',
        'revenue planning',
        'customer onboarding',
        'training',
        'consulting',
        'solution selling'
    ],
    'tools': [
        'salesforce',
        'gainsight',
        'totango',
        'churnzero',
        'clientsuccess',
        'planhat',
        'catalyst',
        'vitally',
        'hubspot',
        'tableau',
        'looker',
        'excel',
        'jira',
        'confluence',
        'slack'
    ],
    'metrics': [
        'arr',
        'mrr',
        'nrr',
        'net revenue retention',
        'gross revenue retention',
        'grr',
        'churn rate',
        'expansion revenue',
        'upsell rate',
        'cross-sell rate',
        'time to value',
        'ttv',
        'product adoption rate',
        'customer health score',
        'qbr effectiveness'
    ]
}

def get_role_patterns(role_type: Optional[str] = None) -> Dict:
    """Get role-specific patterns based on role type.
    
    Args:
        role_type: Optional role type (cx/cs)
        
    Returns:
        Dict containing role-specific patterns
    """
    if role_type:
        role_type = role_type.lower()
        if role_type == 'cx':
            return CX_PATTERNS
        elif role_type == 'cs':
            return CS_PATTERNS
    return {}

def get_role_specific_skills(role_type: Optional[str] = None) -> Set[str]:
    """Get role-specific skills based on role type.
    
    Args:
        role_type: Optional role type (cx/cs)
        
    Returns:
        Set of role-specific skills
    """
    patterns = get_role_patterns(role_type)
    skills = set()
    if patterns:
        skills.update(patterns.get('skills', []))
        skills.update(patterns.get('tools', []))
    return skills

def get_role_specific_metrics(role_type: Optional[str] = None) -> Set[str]:
    """Get role-specific metrics based on role type.
    
    Args:
        role_type: Optional role type (cx/cs)
        
    Returns:
        Set of role-specific metrics
    """
    patterns = get_role_patterns(role_type)
    return set(patterns.get('metrics', []))

def get_role_specific_keywords(role_type: Optional[str] = None) -> Set[str]:
    """Get role-specific keywords based on role type.
    
    Args:
        role_type: Optional role type (cx/cs)
        
    Returns:
        Set of role-specific keywords
    """
    patterns = get_role_patterns(role_type)
    return set(patterns.get('keywords', []))

def detect_role_type(text: str) -> Optional[str]:
    """Detect role type from text using pattern matching.
    
    Args:
        text: Text to analyze
        
    Returns:
        Detected role type (cx/cs) or None
    """
    text = text.lower()
    
    # Count matches for each role type
    cx_matches = sum(1 for keyword in CX_PATTERNS['keywords'] if keyword.lower() in text)
    cs_matches = sum(1 for keyword in CS_PATTERNS['keywords'] if keyword.lower() in text)
    
    # Return role type with more matches
    if cx_matches > cs_matches:
        return 'cx'
    elif cs_matches > cx_matches:
        return 'cs'
    return None

def calculate_role_specific_score(text: str, role_type: Optional[str] = None) -> float:
    """Calculate role-specific score based on keyword matches.
    
    Args:
        text: Text to analyze
        role_type: Optional role type (cx/cs)
        
    Returns:
        Role-specific score between 0 and 1
    """
    if not role_type:
        return 0.0
        
    patterns = get_role_patterns(role_type)
    if not patterns:
        return 0.0
        
    text = text.lower()
    total_matches = 0
    total_patterns = 0
    
    # Check matches across all pattern types
    for pattern_type in ['keywords', 'skills', 'tools', 'metrics']:
        patterns_list = patterns.get(pattern_type, [])
        total_patterns += len(patterns_list)
        total_matches += sum(1 for pattern in patterns_list if pattern.lower() in text)
    
    return total_matches / max(1, total_patterns)
