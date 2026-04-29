import os
from pathlib import Path

class Settings:
    """
    Central configuration for Aether paths and environment settings.
    Ensures the project remains self-contained.
    """
    def __init__(self):
        # Resolve the root directory (d:/Project/Projects/Aether)
        self.project_root = Path(__file__).parent.parent.resolve()
        
        # Define project-specific directories
        self.data_dir = self.project_root / "data"
        self.temp_dir = self.project_root / "temp"
        self.logs_dir = self.project_root / "logs"
        
    def ensure_dirs(self):
        """Create required directories if they don't exist."""
        for path in [self.data_dir, self.temp_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)

settings = Settings()
