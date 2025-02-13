"""Test NLTK initialization and utilities."""
import pytest
import nltk
from app.utils.nltk_utils import ensure_nltk_data, REQUIRED_NLTK_DATA, verify_nltk_data

def test_nltk_data_initialization():
    """Test that NLTK data is properly initialized."""
    # First ensure data is downloaded
    ensure_nltk_data()
    
    # Verify each required resource is available
    for resource_path, package_name in REQUIRED_NLTK_DATA:
        assert verify_nltk_data(resource_path, package_name), f"Required NLTK resource not found: {package_name}"
            
def test_nltk_imports():
    """Test that NLTK components can be imported and used."""
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Test basic functionality
    text = "Testing NLTK functionality"
    
    # Test tokenization
    tokens = word_tokenize(text)
    assert len(tokens) > 0, "Tokenization failed"
    
    # Test stopwords
    stop_words = set(stopwords.words('english'))
    assert len(stop_words) > 0, "Stopwords not loaded"
    
    # Test lemmatization
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize("testing")
    assert isinstance(lemma, str), "Lemmatization failed"

def test_error_handling():
    """Test error handling in NLTK initialization."""
    # Test with invalid resource
    invalid_resources = [("invalid/resource", "invalid")]
    with pytest.raises(RuntimeError, match="Failed to initialize NLTK"):
        ensure_nltk_data(custom_resources=invalid_resources)
