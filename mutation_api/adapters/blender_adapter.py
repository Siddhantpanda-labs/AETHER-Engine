import bpy
from typing import Optional

from core.types import SceneState, SceneObject, MutationResult
from mutation_api.actions import ActionType, MutationAction
from mutation_api.adapters.base_adapter import MutationAdapter

class BlenderAdapter(MutationAdapter):
    """
    Implementation of the MutationAdapter for Blender.
    Executes actions by translating them into bpy operations.
    """
    
    def __init__(self):
        self._next_id = 1
        
    def _get_object_by_id(self, obj_id: int) -> Optional[bpy.types.Object]:
        for obj in bpy.context.scene.objects:
            if obj.get("aether_id") == obj_id:
                return obj
        return None

    def execute(self, action: MutationAction) -> MutationResult:
        try:
            params = action.parameters or {}
            
            if action.action_type == ActionType.SPAWN_OBJECT:
                obj_type = params.get("type", "cube").lower()
                position = params.get("position", (0.0, 0.0, 0.0))
                
                if obj_type == "cube":
                    bpy.ops.mesh.primitive_cube_add(location=position)
                elif obj_type == "sphere":
                    bpy.ops.mesh.primitive_uv_sphere_add(location=position)
                else:
                    return MutationResult(success=False, message=f"Unsupported object type: {obj_type}")
                
                new_obj = bpy.context.active_object
                new_obj["aether_id"] = self._next_id
                self._next_id += 1
                
                return MutationResult(success=True, data={"id": new_obj["aether_id"]})
                
            elif action.action_type == ActionType.DELETE_OBJECT:
                if action.object_id is None:
                    return MutationResult(success=False, message="Missing object_id")
                
                obj = self._get_object_by_id(action.object_id)
                if not obj:
                    return MutationResult(success=False, message=f"Object {action.object_id} not found")
                
                # Deselect all, select target, delete
                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.delete()
                
                return MutationResult(success=True)
                
            elif action.action_type == ActionType.SET_POSITION:
                if action.object_id is None:
                    return MutationResult(success=False, message="Missing object_id")
                
                obj = self._get_object_by_id(action.object_id)
                if not obj:
                    return MutationResult(success=False, message=f"Object {action.object_id} not found")
                
                pos = params.get("position")
                if not pos:
                    return MutationResult(success=False, message="Missing position parameter")
                    
                obj.location = pos
                return MutationResult(success=True)
                
            elif action.action_type == ActionType.SET_SCALE:
                if action.object_id is None:
                    return MutationResult(success=False, message="Missing object_id")
                
                obj = self._get_object_by_id(action.object_id)
                if not obj:
                    return MutationResult(success=False, message=f"Object {action.object_id} not found")
                
                scale = params.get("scale")
                if not scale:
                    return MutationResult(success=False, message="Missing scale parameter")
                    
                obj.scale = scale
                return MutationResult(success=True)
                
            elif action.action_type == ActionType.TASK_COMPLETE:
                return MutationResult(success=True, message="Task marked complete")
                
            else:
                return MutationResult(success=False, message=f"Unhandled action type: {action.action_type}")
                
        except Exception as e:
            return MutationResult(success=False, message=str(e))

    def get_scene_state(self) -> SceneState:
        state = SceneState()
        for obj in bpy.context.scene.objects:
            obj_id = obj.get("aether_id")
            if obj_id is not None:
                scene_obj = SceneObject(
                    id=obj_id,
                    name=obj.name,
                    object_type=obj.type,
                    position=tuple(obj.location),
                    rotation=tuple(obj.rotation_euler),
                    scale=tuple(obj.scale)
                )
                state.objects[obj_id] = scene_obj
        return state

    def reset(self) -> SceneState:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.context.scene.objects:
            if "aether_id" in obj:
                obj.select_set(True)
                
        if bpy.context.selected_objects:
            bpy.ops.object.delete()
            
        self._next_id = 1
        return self.get_scene_state()

    def render_frame(self) -> bytes:
        # Stubbed out for the vertical slice.
        # Image-based RL requires this, but we'll focus on geometric state first.
        return b""
