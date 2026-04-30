import os
import sys

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from llm.backends.api_backend import OpenAIBackend

class DirectorAgent:
    """
    The High-Level LLM Orchestrator.
    Translates natural language into structured JSON goals for the RL PyTorch workers.
    """
    def __init__(self):
        self.llm = OpenAIBackend()
        
        # This is the "System Prompt" that defines the Director's entire personality and rulebook.
        self.system_prompt = """
        You are the Director Agent for the Aether 3D Construction Framework.
        Your job is to translate human natural language requests into structured JSON goals 
        for a low-level PyTorch Reinforcement Learning agent.
        
        The RL agent's neural network currently understands exactly 3 tasks.
        You must map the user's intent to one of these 3 tasks, and infer the required parameters.
        
        1. 'linear_row'
           - Requires: "num_objects" (int), "spacing" (float, usually 2.5), "axis" ("X" or "Y")
        
        2. 'grid'
           - Requires: "rows" (int), "cols" (int), "spacing_x" (float, usually 2.5), "spacing_y" (float, usually 2.5)
        
        3. 'stack'
           - Requires: "height" (int), "spacing_z" (float, MUST be 2.0)
        
        You must ONLY output valid JSON. Do not include markdown formatting or extra text.
        
        Examples:
        User: "Build a 3x3 floor"
        Output: {"task": "grid", "rows": 3, "cols": 3, "spacing_x": 2.1, "spacing_y": 2.1}
        
        User: "Stack 5 cubes on top of each other"
        Output: {"task": "stack", "height": 5, "spacing_z": 2.0}
        
        User: "Place 4 blocks in a line"
        Output: {"task": "linear_row", "num_objects": 4, "spacing": 2.5, "axis": "X"}
        """
        
    def decompose_prompt(self, user_prompt: str) -> dict:
        """Translates English into a mathematical Goal Dictionary."""
        print(f"\n[Director] Listening: '{user_prompt}'")
        print("[Director] Thinking...")
        goal_json = self.llm.generate_json(self.system_prompt, user_prompt)
        print(f"[Director] Decomposed Goal -> {goal_json}")
        return goal_json

if __name__ == "__main__":
    print("\n" + "="*60)
    print("--- AETHER: LLM DIRECTOR AGENT TEST ---")
    print("="*60)
    
    try:
        director = DirectorAgent()
        
        test_prompts = [
            "I need a 4 by 4 tile layout for a room floor.",
            "Make a tall tower, maybe 6 blocks high.",
            "Just put three cubes next to each other on the Y axis."
        ]
        
        for prompt in test_prompts:
            director.decompose_prompt(prompt)
            print("-" * 60)
            
    except Exception as e:
        print(f"\n[Error] Director failed to initialize: {e}")
