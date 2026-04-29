import json
import time
from pathlib import Path
from core.config import settings
from mutation_api.actions import MutationAction

class DemonstrationRecorder:
    """
    Records a sequence of MutationActions into a JSONL (JSON Lines) file.
    This creates the 'demonstration' dataset used for Behavior Cloning.
    """
    
    def __init__(self, session_name: str = None):
        # Default to a timestamped session name
        self.session_name = session_name or f"demo_{int(time.time())}"
        self.filepath = settings.data_dir / f"{self.session_name}.jsonl"
        self.is_recording = False
        
    def start(self):
        """Begin recording actions."""
        self.is_recording = True
        # Ensure data directory exists
        settings.ensure_dirs()
        print(f"[Aether] Started recording demonstration to {self.filepath}")
        
    def stop(self):
        """Stop recording actions."""
        self.is_recording = False
        print(f"[Aether] Stopped recording demonstration.")
        
    def log_action(self, action: MutationAction):
        """Log a single action if recording is active."""
        if not self.is_recording:
            return
            
        # Serialize the action
        action_dict = {
            "timestamp": time.time(),
            "action_type": action.action_type.name,
            "action_value": action.action_type.value,
            "object_id": action.object_id,
            "parameters": action.parameters
        }
        
        # Append as a single JSON line
        with open(self.filepath, "a") as f:
            f.write(json.dumps(action_dict) + "\n")
