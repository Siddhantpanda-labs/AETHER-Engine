import os
import sys
from typing import Tuple, Dict, Any, Optional

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from core.types import SceneState
from mutation_api.actions import ActionType, MutationAction
from mutation_api.adapters.blender_adapter import BlenderAdapter

class BlenderEnv:
    """
    OpenAI Gymnasium-style wrapper for the Blender Engine.
    Allows standard RL algorithms (like PPO) to interact with Blender as a "Game".
    """
    def __init__(self, task=None, max_steps=20):
        # We inject a specific Task (e.g., PlacementTask) to handle Goals and Rewards
        self.task = task
        self.adapter = BlenderAdapter()
        self.current_step = 0
        self.max_steps = max_steps
        
    def reset(self) -> Tuple[SceneState, Dict[str, Any]]:
        """
        Clears the scene and returns the initial state.
        Returns: (observation, info)
        """
        self.current_step = 0
        
        # Tell Blender to delete everything
        self.adapter.reset()
        
        # If we have a task, reset it (e.g., randomize the goal)
        if self.task:
            self.task.reset()
            
        initial_state = self.adapter.get_scene_state()
        info = {"goal": self.task.get_goal() if self.task else {}}
        
        return initial_state, info
        
    def step(self, action: MutationAction) -> Tuple[SceneState, float, bool, bool, Dict[str, Any]]:
        """
        Executes an action in Blender and evaluates the result.
        Returns: (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        info = {}
        
        # 1. Check if AI requested early termination
        if action.action_type == ActionType.TASK_COMPLETE:
            terminated = True
            state = self.adapter.get_scene_state()
            # The task provides the final evaluation reward
            reward = self.task.compute_reward(state, action, is_done=True) if self.task else 0.0
            return state, reward, terminated, False, info
            
        # 2. Execute the action in Blender
        result = self.adapter.execute(action)
        
        if not result.success:
            info["error"] = result.message
            # Penalize the agent for proposing invalid actions
            reward = -0.5
            state = self.adapter.get_scene_state()
            truncated = self.current_step >= self.max_steps
            return state, reward, False, truncated, info
            
        # 3. Observe the new state
        state = self.adapter.get_scene_state()
        
        # 4. Compute reward for this step
        reward = self.task.compute_reward(state, action, is_done=False) if self.task else 0.0
        
        # 5. Check if we hit the step limit (timeout)
        truncated = self.current_step >= self.max_steps
        terminated = False # Only true if TASK_COMPLETE
        
        return state, reward, terminated, truncated, info
