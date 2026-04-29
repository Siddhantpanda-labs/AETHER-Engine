import sys
import os

# Ensure the Aether root directory is in the Python path
# This allows Blender's internal Python to import our aether modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from mutation_api.actions import ActionType, MutationAction
from mutation_api.adapters.blender_adapter import BlenderAdapter
from core.config import settings

def main():
    print("\n" + "="*50)
    print("--- AETHER: VERTICAL SLICE EXECUTION STARTED ---")
    print("="*50 + "\n")
    
    # Ensure project directories exist and tell Blender to use our temp folder
    settings.ensure_dirs()
    if 'bpy' in globals() or 'bpy' in sys.modules:
        import bpy
        bpy.context.preferences.filepaths.temporary_directory = str(settings.temp_dir)
        print(f"Blender temp directory re-routed to: {settings.temp_dir}")

    # Instantiate the adapter we just built
    adapter = BlenderAdapter()
    
    # 1. Clean the default scene (removes default cube, light, camera if needed, 
    # though our reset currently targets only 'aether_id' objects. 
    # For a completely clean slate in Blender, we might want to wipe everything, 
    # but let's stick to our adapter's logic.)
    print("Resetting aether tracked objects...")
    adapter.reset()
    
    # 2. Define the exact sequence to fulfill "place 3 cubes in a row"
    actions = [
        MutationAction(
            action_type=ActionType.SPAWN_OBJECT,
            parameters={"type": "cube", "position": (0.0, 0.0, 0.0)}
        ),
        MutationAction(
            action_type=ActionType.SPAWN_OBJECT,
            parameters={"type": "cube", "position": (3.0, 0.0, 0.0)}
        ),
        MutationAction(
            action_type=ActionType.SPAWN_OBJECT,
            parameters={"type": "cube", "position": (6.0, 0.0, 0.0)}
        ),
        MutationAction(
            action_type=ActionType.TASK_COMPLETE
        )
    ]
    
    # 3. Execute actions through the API layer
    for idx, action in enumerate(actions):
        print(f"Executing: {action.action_type.name}")
        result = adapter.execute(action)
        if result.success:
            print(f"  -> Success! Data: {result.data}")
        else:
            print(f"  -> Failed: {result.message}")

    # 4. Verify by reading back the state
    print("\n--- FINAL SCENE STATE (Read via API) ---")
    state = adapter.get_scene_state()
    for obj_id, obj in state.objects.items():
        # Round the position for cleaner terminal output
        pos = tuple(round(p, 2) for p in obj.position)
        print(f"ID {obj_id} | Type: {obj.object_type} | Position: {pos}")
        
    print("\n" + "="*50)
    print("Vertical Slice Complete!")
    
    # Save the file to our project data folder
    save_path = settings.data_dir / "vertical_slice_result.blend"
    import bpy
    bpy.ops.wm.save_as_mainfile(filepath=str(save_path))
    print(f"Scene saved to: {save_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
