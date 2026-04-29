import torch
import sys
import os
import time

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from core.config import settings
from mutation_api.actions import ActionType, MutationAction
from mutation_api.adapters.blender_adapter import BlenderAdapter
from executor.policy.network import SimpleAetherPolicy

def featurize_state(state):
    """
    Matches the featurization in imitation_trainer.py.
    Extracts object positions and flattens them into a tensor.
    """
    obs_vec = [0.0] * 100 # max 10 objects * 10 features
    for i, (obj_id, obj) in enumerate(state.objects.items()):
        if i >= 10: break
        idx = i * 10
        obs_vec[idx:idx+3] = obj.position
    return torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)

def main():
    print("\n" + "="*60)
    print("--- AETHER: AUTONOMOUS POLICY EVALUATION ---")
    print("="*60 + "\n")

    # Ensure local directories exist and re-route Blender's temp folder
    settings.ensure_dirs()
    if 'bpy' in sys.modules:
        import bpy
        bpy.context.preferences.filepaths.temporary_directory = str(settings.temp_dir)
        print(f"[Aether] Re-routed Blender temp folder to: {settings.temp_dir}")

    # 1. Load the model
    model = SimpleAetherPolicy()
    model_path = settings.data_dir / "bc_policy_v1.pth"
    
    if not model_path.exists():
        print(f"[Error] Model weights not found at {model_path}.")
        print("Please run 'python executor/training/imitation_trainer.py' first.")
        return
        
    # Load weights (ensure we use CPU if no GPU is active in Blender)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[Aether] Loaded policy weights from: {model_path}")

    # 2. Setup the environment
    adapter = BlenderAdapter()
    print("[Aether] Resetting Blender scene for evaluation...")
    adapter.reset()

    # 3. Autonomous Execution Loop
    max_steps = 10
    print(f"[Aether] Starting autonomous build (Max steps: {max_steps})...\n")
    
    for step in range(max_steps):
        # A. Observe the world
        state = adapter.get_scene_state()
        obs_tensor = featurize_state(state).to(device)
        
        # B. Predict the next action
        with torch.no_grad():
            action_logits, _, pred_params = model(obs_tensor)
            
        action_type_idx = torch.argmax(action_logits, dim=1).item()
        action_type = ActionType(action_type_idx)
        
        # Extract predicted parameters (x, y, z)
        params_list = pred_params.squeeze().tolist()
        
        print(f"Step {step+1}:")
        print(f"  - Observation: {len(state.objects)} objects in scene")
        print(f"  - AI Prediction: {action_type.name}")
        
        if action_type == ActionType.TASK_COMPLETE:
            print("\n[Aether] AI has signaled TASK_COMPLETE. Building finished!")
            break
            
        # C. Execute the action
        print(f"  - Parameters: Position {tuple(round(p, 2) for p in params_list)}")
        
        # We'll assume the AI wants to spawn a cube at the predicted position
        action = MutationAction(
            action_type=action_type,
            parameters={"type": "cube", "position": tuple(params_list)}
        )
        
        result = adapter.execute(action)
        if not result.success:
            print(f"  [Error] Execution failed: {result.message}")
            break
            
        # Small delay so you can watch it happen in the viewport
        time.sleep(0.3)

    print("\n" + "="*60)
    print("Evaluation Complete. The AI has finished its construction.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
