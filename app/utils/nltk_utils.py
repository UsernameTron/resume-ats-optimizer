"""NLTK initialization and utilities."""
import logging
import nltk
import os
from typing import List, Tuple

logger = logging.getLogger(__name__)

REQUIRED_NLTK_DATA: List[Tuple[str, str]] = [
    ('tokenizers/punkt', 'punkt'),
    ('corpora/stopwords', 'stopwords'),
    ('wordnet', 'wordnet'),  # Changed path for wordnet
    ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
]

def verify_nltk_data(resource_path: str, package_name: str) -> bool:
    """Verify NLTK resource is properly downloaded and accessible.
    
    Args:
        resource_path: Path to the resource (e.g., 'corpora/wordnet')
        package_name: Name of the package (e.g., 'wordnet')
        
    Returns:
        bool: True if resource is available, False otherwise
    """
    try:
        # Special case for wordnet
        if package_name == 'wordnet':
            from nltk.corpus import wordnet
            # Try to access wordnet data to verify it's working
            synsets = wordnet.synsets('test')
            return len(synsets) > 0
        
        # For other resources, try to find them directly
        nltk.data.find(resource_path)
        return True
    except (LookupError, ImportError, OSError):
        return False

def ensure_nltk_data(custom_resources: List[Tuple[str, str]] = None):
    """Ensure all required NLTK data is downloaded.
    
    This is the single source of truth for NLTK data downloads.
    
    Args:
        custom_resources: Optional list of additional resources to download
        
    Raises:
        RuntimeError: If any required NLTK resource cannot be downloaded or verified
    """
    # Set NLTK data path to project's virtual environment
    venv_nltk_data = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'venv', 'nltk_data')
    os.makedirs(venv_nltk_data, exist_ok=True)
    nltk.data.path.insert(0, venv_nltk_data)
    
    download_errors = []
    resources_to_check = REQUIRED_NLTK_DATA + (custom_resources or [])
    
    for resource_path, package_name in resources_to_check:
        if not verify_nltk_data(resource_path, package_name):
            logger.info(f'Downloading NLTK resource {package_name}')
            try:
                nltk.download(package_name, download_dir=venv_nltk_data)
                if not verify_nltk_data(resource_path, package_name):
                    error_msg = f'Failed to verify {package_name} after download'
                    logger.error(error_msg)
                    download_errors.append(error_msg)
            except Exception as e:
                error_msg = f'Failed to download {package_name}: {str(e)}'
                logger.error(error_msg)
                download_errors.append(error_msg)
        else:
            logger.debug(f'NLTK resource {package_name} already downloaded')
    
    if download_errors:
        raise RuntimeError('Failed to initialize NLTK:\n' + '\n'.join(download_errors))
