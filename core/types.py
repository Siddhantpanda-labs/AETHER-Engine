from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Optional

# Basic geometric types
Vector3 = Tuple[float, float, float]

@dataclass
class SceneObject:
    """
    A representation of an object in the 3D world.
    """
    id: int
    name: str
    object_type: str  # e.g., 'cube', 'sphere'
    position: Vector3 = (0.0, 0.0, 0.0)
    rotation: Vector3 = (0.0, 0.0, 0.0)  # Euler angles in radians
    scale: Vector3 = (1.0, 1.0, 1.0)

@dataclass
class SceneState:
    """
    A snapshot of the current 3D world state.
    This is what the agent observes (along with rendered frames).
    """
    objects: Dict[int, SceneObject] = field(default_factory=dict)
    
@dataclass
class MutationResult:
    """
    The outcome of executing a MutationAction.
    """
    success: bool
    message: str = ""
    # Can carry back data, e.g., the ID of a newly SPAWNed object
    data: Optional[Dict[str, Any]] = None
