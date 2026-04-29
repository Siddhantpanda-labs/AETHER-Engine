from enum import IntEnum
from dataclasses import dataclass
from typing import Any, Dict, Optional

class ActionType(IntEnum):
    """
    The minimal set of actions for our first vertical slice.
    This limits the action space for early stability.
    """
    SPAWN_OBJECT = 0
    DELETE_OBJECT = 1
    SET_POSITION = 2
    SET_SCALE = 3
    TASK_COMPLETE = 4

@dataclass
class MutationAction:
    """
    The universal vocabulary token for all engine operations.
    """
    action_type: ActionType
    
    # Target object ID (None for actions like SPAWN_OBJECT or TASK_COMPLETE)
    object_id: Optional[int] = None
    
    # Action-specific parameters (e.g., {"position": (1.0, 0.0, 0.0)} or {"type": "cube"})
    parameters: Optional[Dict[str, Any]] = None
