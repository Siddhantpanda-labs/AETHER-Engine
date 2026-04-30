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
    Now expanded to handle Sequence 4 Generalization: Grids and Stacking.
    """
    def __init__(self):
        self.output_dir = settings.data_dir / "synthetic"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_all(self, num_per_category: int = 1000):
        self.generate_linear_row(num_per_category)
        self.generate_grid(num_per_category)
        self.generate_stack(num_per_category)
        
    # --- CATEGORY 1: LINEAR ROW ---
    def generate_linear_row(self, num_trajectories: int = 100):
        filepath = self.output_dir / "linear_row_dataset.jsonl"
        with open(filepath, 'w') as f:
            for _ in range(num_trajectories):
                traj = self._generate_single_linear_row()
                f.write(traj.to_json() + "\n")
        print(f"[Aether Data] Generated {num_trajectories} linear_row trajectories at {filepath}")
        
    def _generate_single_linear_row(self) -> Trajectory:
        num_objects = random.randint(2, 6)
        spacing = random.uniform(1.5, 3.5)
        axis = random.choice([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
        axis_name = "X" if axis[0] > 0 else "Y"
        traj_id = f"traj_linear_{uuid.uuid4().hex[:8]}"
        
        goal = {"task": "linear_row", "num_objects": num_objects, "spacing": round(spacing, 2), "axis": axis_name}
        
        positions = []
        for i in range(num_objects):
            pos_x = round(i * spacing * axis[0], 2)
            pos_y = round(i * spacing * axis[1], 2)
            pos_z = 0.0
            positions.append([pos_x, pos_y, pos_z])
            
        return self._build_trajectory(traj_id, goal, positions)

    # --- CATEGORY 2: GRID PLACEMENT ---
    def generate_grid(self, num_trajectories: int = 100):
        filepath = self.output_dir / "grid_dataset.jsonl"
        with open(filepath, 'w') as f:
            for _ in range(num_trajectories):
                traj = self._generate_single_grid()
                f.write(traj.to_json() + "\n")
        print(f"[Aether Data] Generated {num_trajectories} grid trajectories at {filepath}")
        
    def _generate_single_grid(self) -> Trajectory:
        rows = random.randint(2, 4)
        cols = random.randint(2, 4)
        spacing_x = random.uniform(2.0, 3.0)
        spacing_y = random.uniform(2.0, 3.0)
        
        traj_id = f"traj_grid_{uuid.uuid4().hex[:8]}"
        goal = {"task": "grid", "rows": rows, "cols": cols, "spacing_x": round(spacing_x, 2), "spacing_y": round(spacing_y, 2)}
        
        positions = []
        for r in range(rows):
            for c in range(cols):
                pos_x = round(c * spacing_x, 2)
                pos_y = round(r * spacing_y, 2)
                pos_z = 0.0
                positions.append([pos_x, pos_y, pos_z])
                
        return self._build_trajectory(traj_id, goal, positions)
        
    # --- CATEGORY 3: VERTICAL STACK ---
    def generate_stack(self, num_trajectories: int = 100):
        filepath = self.output_dir / "stack_dataset.jsonl"
        with open(filepath, 'w') as f:
            for _ in range(num_trajectories):
                traj = self._generate_single_stack()
                f.write(traj.to_json() + "\n")
        print(f"[Aether Data] Generated {num_trajectories} stack trajectories at {filepath}")

    def _generate_single_stack(self) -> Trajectory:
        height = random.randint(3, 7)
        # Blender default cubes are 2 units tall, so perfect stacking is exactly 2.0 increments
        spacing_z = 2.0  
        
        traj_id = f"traj_stack_{uuid.uuid4().hex[:8]}"
        goal = {"task": "stack", "height": height, "spacing_z": spacing_z}
        
        positions = []
        for z in range(height):
            pos_x = 0.0
            pos_y = 0.0
            pos_z = round(z * spacing_z, 2)
            positions.append([pos_x, pos_y, pos_z])
            
        return self._build_trajectory(traj_id, goal, positions)

    # --- COMMON TRAJECTORY BUILDER ---
    def _build_trajectory(self, traj_id: str, goal: dict, positions: list) -> Trajectory:
        """Helper to convert a list of positions into full MDP sequences."""
        current_objects = []
        current_state = Observation(step_index=0, objects=list(current_objects))
        steps = []
        
        for i, pos in enumerate(positions):
            action = ActionRecord(
                action_type=ActionType.SPAWN_OBJECT.value,
                target_object_id=None,
                parameters={"type": "cube", "position": pos}
            )
            
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
                objects=list(current_objects)
            )
            
            transition = Transition(
                t=i,
                observation=current_state,
                action=action,
                next_observation=next_state,
                reward=0.0,
                done=False
            )
            
            steps.append(transition)
            current_state = next_state
            
        # Final TASK_COMPLETE action
        complete_action = ActionRecord(
            action_type=ActionType.TASK_COMPLETE.value,
            target_object_id=None,
            parameters={}
        )
        
        final_transition = Transition(
            t=len(positions),
            observation=current_state,
            action=complete_action,
            next_observation=current_state,
            reward=1.0,
            done=True
        )
        steps.append(final_transition)
            
        return Trajectory(
            trajectory_id=traj_id,
            version=1,
            task_type=goal["task"],
            goal=goal,
            initial_state=Observation(step_index=0, objects=[]),
            steps=steps,
            success=True,
            total_steps=len(steps)
        )

if __name__ == "__main__":
    generator = ProceduralGenerator()
    # Generate the generalized dataset
    generator.generate_all(num_per_category=1000)
