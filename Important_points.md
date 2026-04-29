
# 3. Where You Need to Be Careful (Important)

Now the critical part — where this can break.

---

## 3.1 RL Complexity Is Extremely High

Your executor is:

* multi-modal
* transformer-based
* continuous + discrete outputs
* operating in a huge action space

This is **very hard to train**.

Risk:

* training instability
* reward collapse
* slow convergence

### Recommendation

Start with:

* **very constrained task space**
* reduce action vocabulary initially
* simplify observation

Example:

* only `SPAWN_OBJECT + SET_POSITION`
* no materials, no lighting initially

Then expand.

---

## 3.2 Action Space Explosion

Your Mutation API is powerful — but:

> too large action space = harder RL

You currently have ~30+ actions.

That’s fine long-term, but:

### Recommendation

Introduce:

* **action masking**
* or **task-specific action subsets**

Otherwise:

* policy struggles to learn meaningful behavior

---

## 3.3 World State Consistency

You rely on:

* Redis (hot)
* Postgres (persistent)

Risk:

* sync issues
* stale reads
* race conditions between agents

### Recommendation

You need:

* **strict write authority**

  * only executor mutates
* versioned state snapshots
* possibly event sourcing (append-only log)

---

## 3.4 Director Agent Dependency on LLM

Your system heavily depends on:

* LLM for decomposition
* LLM for reasoning

Risk:

* latency
* cost
* inconsistency

### Recommendation

Gradually:

* replace frequent decisions with learned policies
* keep LLM for:

  * high-level planning only

---

## 3.5 Blender Bottleneck

Blender is:

* single-threaded for many operations
* not designed for RL throughput

Risk:

* slow training loop

### Recommendation

Eventually:

* build **lightweight simulation backend**
* or parallel Blender instances

---

# 4. What This System Actually Is (Positioning)

This is NOT just:

* a game tool
* a Blender automation system

This is closer to:

> **A general-purpose 3D world construction AI framework**

Comparable directions:

* robotics simulation learning systems
* generative environment builders
* procedural content AI engines

---

# 5. Strategic Strength

Your strongest architectural idea:

> **AI operates on a universal action language (Mutation API)**

That unlocks:

* engine independence
* training-production separation
* cross-domain reuse

This is the **core intellectual asset of your system**.

---

# 6. What I Would Do Next (Precise Steps)

Do not expand architecture further.

Instead:

### Step 1 — Make It Real (Minimal Vertical Slice)

Implement:

* 5 actions only:

  * SPAWN_OBJECT
  * SET_POSITION
  * SET_SCALE
  * DELETE_OBJECT
  * TASK_COMPLETE

* 1 task:

  * “place 3 cubes in a row”

* no agents initially

* no Ray initially

Just:

* Blender env
* Mutation API
* simple loop

---

### Step 2 — Add Learning (BC only)

* record manual actions
* train imitation model

No RL yet.

---

### Step 3 — Add RL (small scope)

* single task
* reward = correct placement

---

### Step 4 — Introduce Director + Agents

Only after executor works.

---

# 7. Final Verdict

**This is not a beginner system.**
This is already at the level of:

* early-stage research lab architecture
* or a serious startup prototype

But:

> The risk is not bad design —
> the risk is *trying to build all of it at once*.

---
