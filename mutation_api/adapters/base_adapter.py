from typing import Protocol
from core.types import SceneState, MutationResult
from mutation_api.actions import MutationAction

class MutationAdapter(Protocol):
    """
    The load-bearing interface for engine adapters.
    Whether Blender (bpy) or C++ engine, they must implement this.
    """
    
    def execute(self, action: MutationAction) -> MutationResult:
        """Apply a mutation action to the engine."""
        ...

    def get_scene_state(self) -> SceneState:
        """Read the current engine state into the stable SceneState format."""
        ...

    def reset(self) -> SceneState:
        """Clear the environment for a new episode/session."""
        ...

    def render_frame(self) -> bytes:
        """Return the current visual observation as image bytes (e.g., PNG)."""
        ...
