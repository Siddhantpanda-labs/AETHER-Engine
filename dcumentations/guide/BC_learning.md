# 1. What BC Actually Needs in *Your System*

From your architecture, BC is not “learning Blender actions.”
It is learning:

> **Mapping: (Scene State, Goal Context) → MutationAction sequence**

Because your executor is a **policy network over Mutation API**, not over Blender UI. 

So your BC dataset must teach:

* Action sequencing
* Spatial reasoning
* Object relationships
* Parameter prediction (continuous outputs)

Not just “spawn cube”.

---

## Therefore, BC data must cover:

### A. Atomic Skills (low-level primitives)

These map directly to action heads in your policy:

* SPAWN_OBJECT
* SET_POSITION
* SET_ROTATION
* SET_SCALE
* DELETE_OBJECT

These are your **action vocabulary grounding layer**

---

### B. Compositional Skills (multi-step sequences)

This is where learning actually happens:

* “Place 3 cubes in a row”
* “Stack objects vertically”
* “Create grid”
* “Align objects”

These teach:

* sequencing
* relational reasoning
* parameter consistency

---

### C. Structured Tasks (goal-driven trajectories)

These align with your environment task suite:

* spawn_task
* room_task
* placement_task 

Examples:

* “Create a wall”
* “Build a simple room”
* “Arrange objects in layout”

These are **critical**, because:

> BC must approximate the distribution of tasks RL will later optimize.

---

# 2. What Synthetic Logs Should Look Like (Critical)

Your logs must match:

```
(state_t, action_t, state_t+1)
```

Because your pipeline has:

* observation encoder
* policy network
* replay buffer 

---

## Correct trajectory format (conceptual)

Each demonstration:

```
Trajectory:
  initial_state
  goal (optional but recommended)

  steps:
    - obs_0 → action_0
    - obs_1 → action_1
    - ...
    - obs_n → action_n

  terminal_flag
```

---

## Important: Do NOT generate only action lists

Wrong:

```
spawn → move → scale
```

Correct:

```
(state → action → next_state)
```

Because:

> Your model is conditional on state, not sequence index.

---

# 3. What Synthetic Data You Should Generate

Now the important part.

You should NOT randomly generate actions.

You should generate **task-distribution-aligned trajectories**.

---

## Phase 1 Dataset Categories (Minimal Viable BC)

### Category 1 — Linear Placement

* 2–10 objects in a row
* variable spacing
* random axis (x/y/z)

Purpose:

* teach spatial consistency
* teach parameter regression

---

### Category 2 — Grid Placement

* NxM grids (2x2 → 5x5)

Purpose:

* structured layout understanding

---

### Category 3 — Stacking / Vertical Structures

* towers
* pyramids

Purpose:

* z-axis reasoning

---

### Category 4 — Transform Sequences

* spawn → move → rotate → scale

Purpose:

* multi-head coordination (your policy has 3 heads) 

---

### Category 5 — Room Construction (VERY IMPORTANT)

* floor
* 4 walls

Purpose:

* first real “scene-level” skill
* aligns with `room_task`

---

### Category 6 — Object Arrangement

* place objects relative to each other

Purpose:

* relational understanding

---

# 4. How Much Data You Actually Need

This is where most people go wrong.

You do NOT need millions initially.

---

## Phase 1 Target (practical and sufficient)

Start with:

### Total: **20,000 – 50,000 trajectories**

Breakdown:

| Category            | Count |
| ------------------- | ----- |
| Linear placement    | 5,000 |
| Grid placement      | 5,000 |
| Stacking            | 5,000 |
| Transform sequences | 5,000 |
| Room construction   | 5,000 |
| Object arrangement  | 5,000 |

---

## Why this is enough

Because:

* Synthetic data = noise-free
* High consistency = faster convergence
* Your action space is structured (not continuous chaos)

---

## Important refinement

Within each trajectory:

* 5–30 steps
* randomized parameters
* small noise injection

Example:

* spacing: 1.0 ± 0.2
* rotation: ±5°

This prevents memorization.

---

# 5. Critical Design Insight (Most Important Part)

You should NOT treat BC as final training.

In your architecture:

```
Phase 1 → BC
Phase 2 → PPO
Phase 3 → World Model
```



BC is only:

> **policy initialization**

So the goal is:

* not perfection
* but *structured prior*

---

# 6. Strategic Recommendation (Very Important)

Yes — you should go fully synthetic first.

And specifically:

* Skip Blender UI
* Skip manual recording
* Build `procedural_generator.py`

Because your architecture already anticipates this. 

---

# 7. Final Answer (Condensed)

**What kind of BC?**

* State → Action trajectories
* Task-aligned (not random)
* Cover primitives + compositions + structured tasks

**How much?**

* 20k–50k trajectories
* balanced across 5–6 task categories

**Core principle:**

> Teach structure, not exact scenes.

---

> A **strict, serializable, versioned data structure** that represents one full demonstration episode in a form directly consumable by your policy network.

This must align exactly with:

* `core/types.py`
* `MutationAction`
* `SceneState`
* observation encoder
* replay buffer format 

---

# 1. First Principles — What Must Be Represented

Given your architecture, each step must contain:

* **Observation (state encoding input)**
* **Action (supervision target)**
* **Result (optional but useful for validation)**
* **Done flag**

Because your policy learns:

> π(a | state)

---

# 2. Canonical Trajectory Schema (Production-Grade)

Below is the exact structure you should implement.

## Top-Level Trajectory

```python
# data/schema.py

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class Trajectory:
    trajectory_id: str
    version: int

    # Task context (VERY IMPORTANT for generalization)
    task_type: str                  # "linear_row", "grid", "room", etc.
    goal: Dict[str, Any]            # structured goal description

    # Initial environment state
    initial_state: "SceneState"

    # Sequence of transitions
    steps: List["Transition"]

    # Terminal info
    success: bool
    total_steps: int
```

---

## Transition (Core Learning Unit)

```python
@dataclass
class Transition:
    t: int

    observation: "Observation"
    action: "ActionRecord"

    next_observation: "Observation"

    reward: float                  # 0 for BC, used later for RL
    done: bool
```

---

# 3. Observation Schema (CRITICAL — Matches Your Encoder)

Your executor has:

* object encoder
* spatial encoder
* visual encoder 

So your observation must match that.

---

## Observation

```python
@dataclass
class Observation:
    # Object-level structured data
    objects: List["ObjectState"]

    # Optional spatial representation
    voxel_grid: Optional[Any]      # compressed or omitted in Phase 1

    # Optional visual input
    image: Optional[bytes]         # PNG or JPEG

    # Metadata
    step_index: int
```

---

## ObjectState

```python
@dataclass
class ObjectState:
    object_id: int
    object_type: str              # "cube", "light", etc.

    position: List[float]         # [x, y, z]
    rotation: List[float]         # [rx, ry, rz]
    scale: List[float]            # [sx, sy, sz]

    parent_id: Optional[int]

    # Optional extensions
    material: Optional[str]
```

---

# 4. Action Schema (Aligned with Mutation API)

This must map **1:1** with your `ActionType` enum. 

---

## ActionRecord

```python
@dataclass
class ActionRecord:
    action_type: int              # maps to ActionType enum

    target_object_id: Optional[int]

    parameters: Dict[str, Any]    # flexible but structured
```

---

## Example Actions

### Spawn Object

```python
ActionRecord(
    action_type=ActionType.SPAWN_OBJECT,
    target_object_id=None,
    parameters={
        "object_type": "cube",
        "position": [0, 0, 0]
    }
)
```

---

### Set Position

```python
ActionRecord(
    action_type=ActionType.SET_POSITION,
    target_object_id=3,
    parameters={
        "position": [2.0, 0.0, 0.0]
    }
)
```

---

# 5. Why This Structure Is Non-Negotiable

Because your policy architecture explicitly expects:

* discrete head → `action_type`
* pointer head → `target_object_id`
* regression head → `parameters` 

If your schema does not align with this:

* training becomes unstable
* heads learn inconsistent mappings
* policy collapses during RL

---

# 6. Minimal JSON Representation (for Storage)

You will not store dataclasses directly. Use JSON.

Example:

```json
{
  "trajectory_id": "traj_0001",
  "version": 1,
  "task_type": "linear_row",
  "goal": {
    "num_objects": 3,
    "spacing": 2.0,
    "axis": "x"
  },
  "initial_state": {
    "objects": []
  },
  "steps": [
    {
      "t": 0,
      "observation": {
        "objects": []
      },
      "action": {
        "action_type": 0,
        "target_object_id": null,
        "parameters": {
          "object_type": "cube"
        }
      },
      "next_observation": {
        "objects": [
          {
            "object_id": 1,
            "object_type": "cube",
            "position": [0,0,0],
            "rotation": [0,0,0],
            "scale": [1,1,1]
          }
        ]
      },
      "reward": 0,
      "done": false
    }
  ],
  "success": true,
  "total_steps": 10
}
```

---

# 7. Important Design Decisions (Do NOT Ignore)

## 1. Include `next_observation`

Even for BC.

Reason:

* required for replay buffer
* required for PPO later
* avoids recomputation

---

## 2. Include `goal`

This is what allows:

> generalization across tasks

Without it, your model learns:

* “blind action sequences”
  instead of:
* “goal-conditioned behavior”

---

## 3. Keep schema versioned

```python
version: int
```

You will change it later. Guaranteed.

---

## 4. Keep parameters flexible but structured

Avoid:

```python
"params": [1,2,3]
```

Use:

```python
"position": [x,y,z]
```

This improves learnability.

---

# 8. Phase 1 Simplification (Recommended)

To move fast:

* Skip voxel_grid
* Skip image

Use only:

```python
objects → structured state
```

This is sufficient for BC bootstrap.

---

# 9. Final Conceptual Summary

A trajectory is:

> A **goal-conditioned sequence of state-action transitions**, fully aligned with your Mutation API and policy heads.

---
