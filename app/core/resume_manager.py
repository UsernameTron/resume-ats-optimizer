import json
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class ResumeManager:
    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path.home() / "CascadeProjects" / "enhanced-ats-optimizer" / "data"
        self.data_dir = data_dir
        self.resume_file = self.data_dir / "stored_resume.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_resume(self, resume_text: str, metadata: Optional[Dict] = None) -> bool:
        """Save resume text and optional metadata."""
        try:
            data = {
                "resume_text": resume_text,
                "metadata": metadata or {}
            }
            with open(self.resume_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Resume saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving resume: {e}")
            return False

    def get_stored_resume(self) -> Optional[Dict]:
        """Get the stored resume and metadata."""
        try:
            if not self.resume_file.exists():
                logger.info("No stored resume found")
                return None
            
            with open(self.resume_file, 'r') as f:
                data = json.load(f)
            logger.info("Resume loaded successfully")
            return data
        except Exception as e:
            logger.error(f"Error loading resume: {e}")
            return None
            
    # Alias for backward compatibility
    load_resume = get_stored_resume

    def update_metadata(self, metadata: Dict) -> bool:
        """Update resume metadata while preserving the resume text."""
        try:
            current_data = self.load_resume()
            if current_data is None:
                logger.error("No resume found to update metadata")
                return False
            
            current_data["metadata"] = {**current_data.get("metadata", {}), **metadata}
            return self.save_resume(current_data["resume_text"], current_data["metadata"])
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            return False

    def clear_stored_resume(self) -> bool:
        """Remove stored resume."""
        try:
            if self.resume_file.exists():
                self.resume_file.unlink()
                logger.info("Resume cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing resume: {e}")
            return False
