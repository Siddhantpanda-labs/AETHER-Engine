import uuid
import random
import os
import sys

# Ensure the project root is in the python path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from core.config import settings
from data.schema import Trajectory, Transition, Observation, ObjectState, ActionRecord
from mutation_api.actions import ActionType

class ProceduralGenerator:
    """
    Generates pure, mathematically correct synthetic demonstrations.
    This creates the Phase 1 BC dataset without requiring a human or GUI.
    """
    def __init__(self):
        self.output_dir = settings.data_dir / "synthetic"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_linear_row(self, num_trajectories: int = 100):
        """
        Category 1: Linear Placement.
        Generates trajectories for placing N cubes in a row along a random axis.
        """
        filepath = self.output_dir / "linear_row_dataset.jsonl"
        
        with open(filepath, 'w') as f:
            for _ in range(num_trajectories):
                traj = self._generate_single_linear_row()
                f.write(traj.to_json() + "\n")
                
        print(f"[Aether Data] Generated {num_trajectories} synthetic trajectories at {filepath}")
        
    def _generate_single_linear_row(self) -> Trajectory:
        # 1. Randomize task parameters to prevent policy memorization
        num_objects = random.randint(2, 6)
        spacing = random.uniform(1.5, 3.5)
        
        # Randomly choose axis: X (1,0,0) or Y (0,1,0)
        axis = random.choice([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
        axis_name = "X" if axis[0] > 0 else "Y"
        
        traj_id = f"traj_linear_{uuid.uuid4().hex[:8]}"
        
        goal = {
            "task": "linear_row",
            "num_objects": num_objects,
            "spacing": round(spacing, 2),
            "axis": axis_name
        }
        
        # 2. Setup initial state
        current_objects = []
        current_state = Observation(step_index=0, objects=list(current_objects))
        steps = []
        
        # 3. Simulate the sequence of actions and state transitions
        for i in range(num_objects):
            # Calculate where the expert would place the next cube
            pos_x = round(i * spacing * axis[0], 2)
            pos_y = round(i * spacing * axis[1], 2)
            pos_z = 0.0
            pos = [pos_x, pos_y, pos_z]
            
            # Action: Expert decides to spawn a cube here
            action = ActionRecord(
                action_type=ActionType.SPAWN_OBJECT.value,
                target_object_id=None,
                parameters={"type": "cube", "position": pos}
            )
            
            # State Update: What the world looks like after the action
            new_obj = ObjectState(
                object_id=i + 1,
                object_type="cube",
                position=pos,
                rotation=[0.0, 0.0, 0.0],
                scale=[1.0, 1.0, 1.0]
            )
            current_objects.append(new_obj)
            
            next_state = Observation(
                step_index=i + 1,
                objects=list(current_objects) # deep copy trick for simple lists
            )
            
            # Create Transition (s, a, s', r, done)
            transition = Transition(
                t=i,
                observation=current_state,
                action=action,
                next_observation=next_state,
                reward=0.0, # BC doesn't use rewards yet
                done=False
            )
            
            steps.append(transition)
            current_state = next_state
            
        # 4. Final step: The expert declares the task complete
        complete_action = ActionRecord(
            action_type=ActionType.TASK_COMPLETE.value,
            target_object_id=None,
            parameters={}
        )
        
        final_transition = Transition(
            t=num_objects,
            observation=current_state,
            action=complete_action,
            next_observation=current_state, # state doesn't change
            reward=1.0, # successful termination
            done=True
        )
        steps.append(final_transition)
            
        # 5. Assemble Trajectory
        return Trajectory(
            trajectory_id=traj_id,
            version=1,
            task_type="linear_row",
            goal=goal,
            initial_state=Observation(step_index=0, objects=[]),
            steps=steps,
            success=True,
            total_steps=len(steps)
        )

if __name__ == "__main__":
    generator = ProceduralGenerator()
    # Generate 1000 trajectories as a starting test
    generator.generate_linear_row(num_trajectories=1000)
