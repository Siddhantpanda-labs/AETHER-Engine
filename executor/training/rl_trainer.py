import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.config import settings
from environment.blender_env import BlenderEnv
from environment.task_suite.placement_task import PlacementTask
from executor.policy.network import SimpleAetherPolicy
from mutation_api.actions import ActionType, MutationAction

def featurize_state(state):
    """Flatten observation matching the BC phase."""
    obs_vec = [0.0] * 100
    for i, (obj_id, obj) in enumerate(state.objects.items()):
        if i >= 10: break
        idx = i * 10
        obs_vec[idx:idx+3] = obj.position
    return torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)

class RLTrainer:
    """
    Executes Policy Gradient (REINFORCE/PPO skeleton) to fine-tune the 
    pre-trained Behavior Cloning policy against the mathematical reward function.
    """
    def __init__(self):
        # 1. Setup the Gym Environment with our Placement Task
        self.task = PlacementTask()
        self.env = BlenderEnv(task=self.task, max_steps=10)
        
        # 2. Load the "Prior" Policy we trained in Phase 1
        self.policy = SimpleAetherPolicy()
        bc_path = settings.data_dir / "bc_policy_v1.pth"
        if bc_path.exists():
            self.policy.load_state_dict(torch.load(bc_path))
            print(f"[Aether RL] Bootstrapped RL from pre-trained BC policy: {bc_path}")
        else:
            print("[Aether RL] Warning: BC policy not found. RL starting from zero knowledge.")
            
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4) # Lower LR for fine-tuning
        
    def train(self, num_episodes=50):
        print("\n" + "="*60)
        print("--- AETHER: REINFORCEMENT LEARNING FINE-TUNING ---")
        print("="*60 + "\n")
        
        for episode in range(num_episodes):
            state, info = self.env.reset()
            goal = info.get("goal", {})
            print(f"Episode {episode+1}: Goal -> Place {goal.get('num_objects')} on {goal.get('axis')} axis (spacing: {goal.get('spacing')})")
            
            log_probs = []
            rewards = []
            
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                obs_tensor = featurize_state(state)
                
                # Forward pass
                action_logits, _, pred_params = self.policy(obs_tensor)
                
                # Convert logits to probabilities and sample an action (Exploration)
                action_probs = torch.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                
                action_idx = action_dist.sample()
                log_prob = action_dist.log_prob(action_idx)
                
                action_type = ActionType(action_idx.item())
                params_list = pred_params.squeeze().tolist()
                
                action = MutationAction(
                    action_type=action_type,
                    parameters={"type": "cube", "position": tuple(params_list)}
                )
                
                # Step the environment and observe reward
                next_state, reward, terminated, truncated, step_info = self.env.step(action)
                
                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state
                
            # Calculate Discounted Returns (How good the whole episode was)
            gamma = 0.99
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
                
            returns = torch.tensor(returns)
            # Normalize returns to stabilize training
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                
            # Perform Policy Gradient Update
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R) # Negative because PyTorch minimizes loss
                
            self.optimizer.zero_grad()
            if policy_loss:
                loss = torch.stack(policy_loss).sum()
                loss.backward()
                self.optimizer.step()
                
            total_reward = sum(rewards)
            print(f"  Result: Total Reward = {total_reward:.2f} | Loss = {loss.item() if policy_loss else 0.0:.4f}")

        print("\n[Aether RL] RL Training session complete.")
        
        # Save the fine-tuned expert policy
        save_path = settings.data_dir / "rl_policy_v1.pth"
        torch.save(self.policy.state_dict(), save_path)
        print(f"[Aether RL] Saved fine-tuned expert policy to {save_path}\n")

if __name__ == "__main__":
    trainer = RLTrainer()
    # We will run a quick 5-episode smoke test to prove the loop works
    trainer.train(num_episodes=5)
