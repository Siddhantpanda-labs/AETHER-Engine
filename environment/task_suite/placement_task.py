import random
import math
from typing import Dict, Any
from core.types import SceneState
from mutation_api.actions import ActionType, MutationAction

class PlacementTask:
    """
    Reward logic for the Linear Placement task.
    Evaluates how well the agent places N objects in a line with specific spacing.
    """
    def __init__(self):
        self.goal = {}
        self.reset()
        
    def reset(self):
        """Randomizes a new goal at the start of each episode."""
        axis = random.choice(["X", "Y"])
        self.goal = {
            "task": "linear_row",
            "num_objects": random.randint(2, 5),
            "spacing": round(random.uniform(2.5, 4.0), 2), # >2.0 to avoid physical overlap
            "axis": axis
        }
        
    def get_goal(self) -> Dict[str, Any]:
        return self.goal
        
    def compute_reward(self, state: SceneState, action: MutationAction, is_done: bool) -> float:
        """
        The critical RL math. Calculates a float reward based on scene geometry.
        """
        reward = 0.0
        objects = list(state.objects.values())
        num_present = len(objects)
        target_num = self.goal["num_objects"]
        
        # ---------------------------------------------------------
        # 1. Step-by-Step Shaping Rewards (During the build)
        # ---------------------------------------------------------
        if not is_done:
            if action.action_type == ActionType.SPAWN_OBJECT:
                if num_present > target_num:
                    reward -= 2.0  # Big penalty for ignoring the goal count
                else:
                    reward += 0.5  # Small cookie for taking productive action
            return reward
            
        # ---------------------------------------------------------
        # 2. Final Evaluation (When AI calls TASK_COMPLETE)
        # ---------------------------------------------------------
        
        # Penalty for quitting too early or too late
        if num_present != target_num:
            reward -= 5.0 
            
        # Evaluate precision math if there are enough objects
        if num_present >= 2:
            axis_idx = 0 if self.goal["axis"] == "X" else 1
            # Sort objects purely along the target axis
            objects.sort(key=lambda obj: obj.position[axis_idx])
            
            spacing_error = 0.0
            alignment_error = 0.0
            overlap_penalty = 0.0
            
            for i in range(1, num_present):
                prev = objects[i-1].position
                curr = objects[i].position
                
                # Euclidean distance
                dist = math.sqrt(sum((curr[j] - prev[j])**2 for j in range(3)))
                
                # Overlap check (Blender cubes are 2x2x2 by default)
                if dist < 2.0:
                    overlap_penalty += 3.0 # Severe penalty for physics intersection
                    
                # Spacing precision
                spacing_error += abs(dist - self.goal["spacing"])
                
                # Alignment precision (punish drift on the wrong axes)
                other_axis_idx = 1 if axis_idx == 0 else 0
                alignment_error += abs(curr[other_axis_idx]) + abs(curr[2]) # Z should be 0
                
            # Apply mathematical penalties
            reward -= overlap_penalty
            reward -= spacing_error
            reward -= alignment_error
            
            # THE JACKPOT: Massive success bonus if perfectly built
            if num_present == target_num and overlap_penalty == 0 and spacing_error < 1.0 and alignment_error < 0.5:
                reward += 15.0
                
        return reward
