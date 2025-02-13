from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Hardware optimization settings
DEVICE_SETTINGS = {
    "USE_MPS": True,  # Use Metal Performance Shaders when available
    "BATCH_SIZE": 32,  # Optimized for M4 Pro
    "NUM_WORKERS": 8   # Based on 12-core CPU
}

# Memory thresholds (in percentage)
MEMORY_THRESHOLDS = {
    "WARNING": 70.0,    # 33.6GB of 48GB
    "CRITICAL": 85.0,   # 40.8GB of 48GB
    "ROLLBACK": 90.0    # 43.2GB of 48GB
}

# Error rate thresholds (in percentage)
ERROR_THRESHOLDS = {
    "WARNING": 5.0,
    "CRITICAL": 10.0,
    "INTERVENTION": 15.0
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "MAX_EXECUTION_TIME": 2.5,  # seconds
    "WARNING_DEGRADATION": 25.0,  # percentage
    "CRITICAL_DEGRADATION": 40.0,
    "ROLLBACK_DEGRADATION": 50.0
}

# Analysis weights
ANALYSIS_WEIGHTS = {
    "SKILLS": 0.4,
    "EXPERIENCE": 0.3,
    "INDUSTRY": 0.2,
    "LOCATION": 0.05,
    "SALARY": 0.05
}

# Minimum scores for success
MINIMUM_SCORES = {
    "ATS_SCORE": 0.8,
    "KEYWORD_DENSITY": 4.5,  # percentage
    "INDUSTRY_MATCH": 0.7
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": BASE_DIR / "logs" / "app.log",
            "formatter": "standard",
            "level": "DEBUG"
        }
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    }
}
