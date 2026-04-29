import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.config import settings
from executor.policy.network import SimpleAetherPolicy

class BC_TrajectoryDataset(Dataset):
    """
    Parses JSONL trajectory files and converts them into PyTorch tensors.
    """
    def __init__(self, filepath):
        self.transitions = []
        
        with open(filepath, 'r') as f:
            for line in f:
                traj = json.loads(line)
                for step in traj["steps"]:
                    # 1. Parse Observation (Flattened for our simple network)
                    obs_vec = [0.0] * 100 # max 10 objects * 10 features
                    for i, obj in enumerate(step["observation"]["objects"]):
                        if i >= 10: break
                        idx = i * 10
                        obs_vec[idx:idx+3] = obj["position"]
                        
                    # 2. Parse Target Action
                    action_type = step["action"]["action_type"]
                    pos_params = step["action"]["parameters"].get("position", [0.0, 0.0, 0.0])
                    
                    self.transitions.append({
                        "obs": torch.tensor(obs_vec, dtype=torch.float32),
                        "action_type": torch.tensor(action_type, dtype=torch.long),
                        "params": torch.tensor(pos_params, dtype=torch.float32)
                    })
                    
    def __len__(self):
        return len(self.transitions)
        
    def __getitem__(self, idx):
        return self.transitions[idx]


def train():
    dataset_path = settings.data_dir / "synthetic" / "linear_row_dataset.jsonl"
    print(f"\n[Aether BC] Loading dataset from {dataset_path}")
    
    try:
        dataset = BC_TrajectoryDataset(dataset_path)
    except FileNotFoundError:
        print("Dataset not found! Run procedural_generator.py first.")
        return
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"[Aether BC] Loaded {len(dataset)} individual (State -> Action) transitions.")
    
    # Initialize our Agent Policy
    model = SimpleAetherPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # We have multiple heads, so we need multiple loss functions
    criterion_classification = nn.CrossEntropyLoss() # For discrete Action Type
    criterion_regression = nn.MSELoss()              # For continuous Parameters
    
    epochs = 10
    print("\n[Aether BC] Starting Imitation Training Loop...\n")
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch in dataloader:
            obs = batch["obs"]
            target_action = batch["action_type"]
            target_params = batch["params"]
            
            optimizer.zero_grad()
            
            # Forward Pass through Policy Network
            action_logits, _, pred_params = model(obs)
            
            # Calculate Loss for each head
            loss_action = criterion_classification(action_logits, target_action)
            loss_params = criterion_regression(pred_params, target_params)
            
            # Combined Loss (Weight the regression loss slightly lower for stability)
            loss = loss_action + (0.1 * loss_params)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1:02d}/{epochs} | Average Combined Loss: {avg_loss:.4f}")
        
    print("\n[Aether BC] Training Complete! The policy network has successfully memorized the mapping.")
    
    # Save the model
    save_path = settings.data_dir / "bc_policy_v1.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[Aether BC] Model weights saved to: {save_path}\n")

if __name__ == "__main__":
    train()
