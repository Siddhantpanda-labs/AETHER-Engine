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

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=40, fill='█', printEnd="\r"):
    """A clean, professional terminal progress bar"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    if iteration == total: 
        print()

class BC_TrajectoryDataset(Dataset):
    """
    Parses JSONL trajectory files and converts them into PyTorch tensors.
    Now supports dynamically loading all datasets from a directory.
    """
    def __init__(self, synthetic_dir):
        self.transitions = []
        
        # Load all .jsonl files in the synthetic directory
        for filename in os.listdir(synthetic_dir):
            if not filename.endswith('.jsonl'):
                continue
                
            filepath = synthetic_dir / filename
            print(f"[Aether BC] Parsing dataset: {filename}...")
            
            with open(filepath, 'r') as f:
                for line in f:
                    traj = json.loads(line)
                    for step in traj["steps"]:
                        # 1. Parse Observation (Flattened logic)
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
    synthetic_dir = settings.data_dir / "synthetic"
    print(f"\n" + "="*60)
    print("--- AETHER: GENERALIZED BEHAVIOR CLONING ---")
    print("="*60)
    print(f"[Aether BC] Searching for datasets in {synthetic_dir}\n")
    
    try:
        dataset = BC_TrajectoryDataset(synthetic_dir)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
        
    if len(dataset) == 0:
        print("No transition data found! Run procedural_generator.py first.")
        return
        
    # Increased batch size to handle the massive new datasets
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print(f"\n[Aether BC] Loaded {len(dataset)} generalized (State -> Action) transitions.")
    
    # Initialize our Agent Policy
    model = SimpleAetherPolicy()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    criterion_classification = nn.CrossEntropyLoss()
    criterion_regression = nn.MSELoss()
    
    epochs = 10
    print("\n[Aether BC] Starting Generalized Imitation Training Loop...\n")
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = len(dataloader)
        
        for i, batch in enumerate(dataloader):
            obs = batch["obs"]
            target_action = batch["action_type"]
            target_params = batch["params"]
            
            optimizer.zero_grad()
            
            action_logits, _, pred_params = model(obs)
            
            loss_action = criterion_classification(action_logits, target_action)
            loss_params = criterion_regression(pred_params, target_params)
            
            loss = loss_action + (0.2 * loss_params)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print Beautiful Progress Bar
            print_progress_bar(i + 1, num_batches, prefix=f'Epoch {epoch+1:02d}/{epochs}', suffix=f'Loss: {loss.item():.4f}', length=40)
            
        avg_loss = total_loss / num_batches
        print(f"  -> Epoch {epoch+1:02d} Complete | Average Combined Loss: {avg_loss:.4f}\n")
        
    print("="*60)
    print("[Aether BC] Training Complete! The policy network has internalized 3D spatial geometry.")
    
    # Save the generalized model
    save_path = settings.data_dir / "bc_policy_generalized_v1.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[Aether BC] Generalized Model weights saved to: {save_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    train()
