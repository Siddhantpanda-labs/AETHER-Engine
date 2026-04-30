# Aether Engine Development Roadmap

This document outlines the strict, professional sequence of work required to scale the Aether 3D AI Construction Framework from its baseline Behavioral Cloning (BC) state into a fully autonomous, Reinforcement Learning-driven, LLM-orchestrated system.

---

## Phase 1: The Foundation (Completed)
*   **Engine Agnosticism:** Built `BlenderAdapter` and `MutationAction` API to decouple AI from UI.
*   **Data Pipeline:** Established the strict MDP data schema (`Trajectory`, `Transition`) for training.
*   **Behavioral Cloning:** Developed the `procedural_generator` and `imitation_trainer`. The PyTorch policy can successfully map raw states to sequence actions, acting as a "structured prior."

---

## Phase 2: Reinforcement Learning (Current Focus)
The goal of this phase is to transition the AI from "mimicking averages" (which causes sloppy placement and overlapping) into an expert agent that learns absolute precision via trial-and-error.

### Sequence 1: The RL Environment (Gym Wrapper)
*   **Objective:** Wrap the `BlenderAdapter` into a standard OpenAI Gymnasium interface.
*   **Key Files:** `environment/blender_env.py`
*   **Requirements:** Must implement `reset()`, `step(action)`, and return `(observation, reward, done, info)`.

### Sequence 2: The Reward Function
*   **Objective:** Write the mathematical evaluation logic for the scene.
*   **Key Files:** `environment/task_suite/placement_task.py`
*   **Requirements:** Must heavily penalize overlapping geometry and reward perfect spacing/alignment according to the given `Goal`.

### Sequence 3: PPO Training Loop
*   **Objective:** Hook the environment and the BC-initialized network up to an RL algorithm.
*   **Key Files:** `executor/training/rl_trainer.py`
*   **Requirements:** Use Proximal Policy Optimization (PPO). The agent will practice building millions of times in headless Blender instances.

### Sequence 4: Task Generalization
*   **Objective:** Expand the RL training curriculum.
*   **Key Files:** `environment/task_suite/*`
*   **Requirements:** Train the agent on 2D Grids, Stacking (Z-axis), and Room/Wall construction.

---

## Phase 3: The Multi-Agent Director
Once the RL agent can flawlessly execute spatial goals, we introduce natural language orchestration.

### Sequence 5: The LLM Director
*   **Objective:** Build the high-level semantic planner.
*   **Key Files:** `llm/gateway.py` and `llm/backends/api_backend.py`
*   **Requirements:** Takes a human prompt ("Build a medieval village") and decomposes it into a list of structural JSON `Goals` for the RL agent to execute sequentially.

---
---

## 📈 Progress Tracker

- [x] **Phase 1: Foundation & BC**
  - [x] Build Engine API (Blender Adapter)
  - [x] Define MDP Data Schema
  - [x] Procedural Generation (Synthetic BC Data)
  - [x] Train Imitation Policy
  - [x] Evaluate Autonomous Execution Loop
- [x] **Phase 2: Reinforcement Learning**
  - [x] Seq 1: RL Environment (Gym Wrapper)
  - [x] Seq 2: Reward Function (Placement Task)
  - [x] Seq 3: PPO Training Loop
  - [x] Seq 4: Task Generalization (Grids, Stacking, Rooms)
- [x] **Phase 3: Multi-Agent Director**
  - [x] Seq 5: LLM Gateway & Decomposition Engine
