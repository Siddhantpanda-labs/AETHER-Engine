import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

@dataclass
class ObjectState:
    """Structured representation of a single object."""
    object_id: int
    object_type: str
    position: List[float]
    rotation: List[float]
    scale: List[float]
    parent_id: Optional[int] = None
    material: Optional[str] = None

@dataclass
class Observation:
    """The multi-modal state input to the policy network."""
    step_index: int
    objects: List[ObjectState] = field(default_factory=list)
    # Excluded for Phase 1 to simplify learning
    voxel_grid: Optional[Any] = None
    image: Optional[bytes] = None

@dataclass
class ActionRecord:
    """The target output of the policy network."""
    action_type: int
    target_object_id: Optional[int]
    parameters: Dict[str, Any]

@dataclass
class Transition:
    """A single step (s, a, s', r, done) mapping."""
    t: int
    observation: Observation
    action: ActionRecord
    next_observation: Observation
    reward: float
    done: bool

@dataclass
class Trajectory:
    """A complete episode/demonstration."""
    trajectory_id: str
    version: int
    task_type: str
    goal: Dict[str, Any]
    initial_state: Observation
    steps: List[Transition] = field(default_factory=list)
    success: bool = False
    total_steps: int = 0
    
    def to_dict(self) -> dict:
        """Serialize to a dictionary."""
        return asdict(self)
        
    def to_json(self) -> str:
        """Serialize the entire trajectory to a JSON string."""
        return json.dumps(self.to_dict())
