import torch
import torch.nn as nn

class SimpleAetherPolicy(nn.Module):
    """
    A simplified version of the Aether Policy Network for our vertical slice.
    Takes flattened object states and predicts actions using 3 heads.
    """
    def __init__(self, max_objects=10, object_feature_dim=10):
        super().__init__()
        # Flattened observation size
        self.obs_dim = max_objects * object_feature_dim
        
        # Shared core representation
        self.fc1 = nn.Linear(self.obs_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # --- The 3 Policy Heads (as specified in architecture) ---
        
        # 1. Action Type Head (Discrete choice over the 5 ActionTypes)
        self.action_type_head = nn.Linear(64, 5)
        
        # 2. Target Object Pointer Head (Which object to manipulate)
        self.target_obj_head = nn.Linear(64, max_objects)
        
        # 3. Parameter Head (Continuous regression for positions/scales)
        self.param_head = nn.Linear(64, 3) 
        
    def forward(self, obs_tensor):
        # Extract features from observation
        x = torch.relu(self.fc1(obs_tensor))
        x = torch.relu(self.fc2(x))
        
        # Branch out to the 3 heads
        action_logits = self.action_type_head(x)
        target_logits = self.target_obj_head(x)
        params = self.param_head(x)
        
        return action_logits, target_logits, params
