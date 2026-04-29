# AETHER — Complete System Architecture
> State-of-the-art AI-driven 3D creation system. Engine-agnostic. Distributed. Continuously learning.

---

## 1. System Philosophy

Four laws that never break:

1. **The AI never knows about Blender or any engine.** All actions flow through the Mutation API. Adapters handle translation.
2. **Compute topology is invisible to AI logic.** Agents and the executor never know which node or GPU they run on. Ray handles placement.
3. **Every interface is stable and versioned.** The Mutation API, Agent contract, and World State contract are the load-bearing seams — they do not change as the system scales.
4. **The LLM provider is a configuration choice, not an architectural one.** Agents never call any LLM directly. All LLM calls go through the LLM Gateway, which routes to local (vLLM / Ollama) or API (OpenAI, Anthropic, Gemini) based on config. Swapping provider requires zero code changes.

---

## 2. Full System Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                        USER INTENT LAYER                         ║
║                "Build me a medieval village"                      ║
╚══════════════════════════════╤═══════════════════════════════════╝
                               │
                               ▼
╔══════════════════════════════════════════════════════════════════╗
║                      ORCHESTRATION LAYER                         ║
║                                                                  ║
║   ┌─────────────────────────────────────────────────────────┐   ║
║   │                   DIRECTOR AGENT                         │   ║
║   │  • Understands intent                                    │   ║
║   │  • Decomposes into ordered subtasks                      │   ║
║   │  • Assigns to specialist agents                          │   ║
║   │  • Monitors overall progress via World State             │   ║
║   │  • Never calls any LLM directly → uses LLM Gateway      │   ║
║   └────────────────────────┬────────────────────────────────┘   ║
║                            │                                      ║
║          ┌─────────────────┼──────────────────┐                  ║
║          ▼                 ▼                  ▼                  ║
║   ┌─────────────┐  ┌─────────────┐  ┌──────────────┐           ║
║   │   LAYOUT    │  │    ASSET    │  │   LIGHTING   │           ║
║   │   AGENT     │  │    AGENT    │  │    AGENT     │           ║
║   │ Spatial     │  │ Object      │  │ Mood/atmos-  │           ║
║   │ composition │  │ selection   │  │ phere design │           ║
║   └──────┬──────┘  └──────┬──────┘  └──────┬───────┘           ║
║          └────────────────┴─────────────────┘                    ║
║                            │ all agents call LLM Gateway          ║
║                            ▼                                      ║
║   ┌─────────────────────────────────────────────────────────┐   ║
║   │                    LLM GATEWAY                           │   ║
║   │          (single abstraction all agents use)             │   ║
║   │                                                          │   ║
║   │   mode = config  ──▶  "api"   or  "local"               │   ║
║   │                                                          │   ║
║   │   ┌──────────────────┐    ┌───────────────────────┐     │   ║
║   │   │   API BACKEND    │    │    LOCAL BACKEND       │     │   ║
║   │   │                  │    │                        │     │   ║
║   │   │  OpenAI (GPT-4o) │    │  vLLM + Ollama         │     │   ║
║   │   │  Anthropic       │    │  Llama 3.1 8B          │     │   ║
║   │   │  Gemini          │    │  Qwen 2.5 / DeepSeek   │     │   ║
║   │   │  Any OpenAI-     │    │  Any HuggingFace model │     │   ║
║   │   │  compatible API  │    │  (quantized, local GPU)│     │   ║
║   │   └──────────────────┘    └───────────────────────┘     │   ║
║   │                                                          │   ║
║   │   + Response cache (Redis) — same intent = no API call   │   ║
║   └─────────────────────────────────────────────────────────┘   ║
║                            │ structured AgentTask                 ║
╚════════════════════════════╪═════════════════════════════════════╝
                             │
                             ▼
╔══════════════════════════════════════════════════════════════════╗
║                       EXECUTION LAYER                            ║
║                                                                  ║
║   ┌─────────────────────────────────────────────────────────┐   ║
║   │                    RL EXECUTOR                           │   ║
║   │                                                          │   ║
║   │  OBSERVATION ENCODER                                     │   ║
║   │  ├── Object list  →  Transformer  → object embeddings   │   ║
║   │  ├── Voxel grid   →  3D CNN       → spatial features    │   ║
║   │  └── Render frame →  ViT / CLIP   → visual features     │   ║
║   │                  ↓ cross-attention fusion                │   ║
║   │  POLICY NETWORK (Transformer decoder)                    │   ║
║   │  ├── Head 1: Action type      (discrete softmax)        │   ║
║   │  ├── Head 2: Object pointer   (attention over embeds)   │   ║
║   │  └── Head 3: Parameters       (continuous regression)   │   ║
║   └────────────────────────┬────────────────────────────────┘   ║
║                            │  MutationAction                      ║
╚════════════════════════════╪═════════════════════════════════════╝
                             │
                             ▼
╔══════════════════════════════════════════════════════════════════╗
║                       MUTATION API LAYER                         ║
║                                                                  ║
║   ┌─────────────────────────────────────────────────────────┐   ║
║   │              MUTATION API  (stable, versioned)           │   ║
║   │   SpawnObject · DeleteObject · SetPosition · SetScale   │   ║
║   │   SetRotation · SetParent · AssignMaterial · SpawnLight │   ║
║   │   SetKeyframe · ApplyModifier · SetWorldProperty · ...  │   ║
║   └──────────────────┬──────────────────┬───────────────────┘   ║
║                      │                  │                         ║
║          ┌───────────▼──────┐  ┌────────▼──────────┐           ║
║          │  BLENDER ADAPTER │  │  ENGINE ADAPTER    │           ║
║          │  (training)      │  │  (pybind11 → C++)  │           ║
║          │  bpy.ops / data  │  │  Your MutationAPI  │           ║
║          └──────────────────┘  └────────────────────┘           ║
╚══════════════════════════════════════════════════════════════════╝
                             │
                             ▼
╔══════════════════════════════════════════════════════════════════╗
║                      SHARED WORLD STATE                          ║
║                                                                  ║
║   Redis (hot)          │  Postgres (persistent)                  ║
║   ├── Scene graph      │  ├── Episodic memory                    ║
║   ├── Task queue       │  ├── Training trajectories              ║
║   ├── Agent states     │  ├── Execution history                  ║
║   └── Working memory   │  └── Demonstration logs                 ║
╚══════════════════════════════════════════════════════════════════╝
                             │
                             ▼
╔══════════════════════════════════════════════════════════════════╗
║                      TRAINING PIPELINE                           ║
║                                                                  ║
║  Phase 1: Behavior Cloning                                       ║
║  Demonstrations → BC Trainer (PyTorch) → Pretrained Policy       ║
║                                                                  ║
║  Phase 2: RL Fine-tuning                                         ║
║  Critic Agent scores → PPO Trainer (TorchRL) → Better Policy     ║
║                                                                  ║
║  Phase 3: World Model                                            ║
║  Dreamer-V3 → Policy trains in latent imagination space          ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 3. Distributed Node System

### Topology

```
╔══════════════════════════════════════════════════════════════════╗
║                      RAY CLUSTER                                 ║
║                                                                  ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │  HEAD NODE — Your RTX 4060 Laptop                        │    ║
║  │                                                           │    ║
║  │  • Ray head process                                       │    ║
║  │  • Director Agent (primary)                               │    ║
║  │  • World State writes (Redis + Postgres)                  │    ║
║  │  • Blender Gym environment                                │    ║
║  │  • Mutation API dispatcher                                │    ║
║  │  • Experiment tracker (W&B)                               │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                                                                  ║
║  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  ║
║  │  WORKER NODE A  │  │  WORKER NODE B  │  │ WORKER NODE C  │  ║
║  │  Laptop GPU     │  │  Laptop GPU     │  │ Laptop GPU     │  ║
║  │                 │  │                 │  │                │  ║
║  │  RL Executor    │  │  Specialist     │  │ Training       │  ║
║  │  (policy fwd    │  │  Agent Pool     │  │ Service        │  ║
║  │   pass +        │  │  Layout/Asset/  │  │ BC + PPO       │  ║
║  │   rollouts)     │  │  Lighting LLMs  │  │ trainer        │  ║
║  └─────────────────┘  └─────────────────┘  └────────────────┘  ║
║                                                                  ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │  CLOUD BURST (on demand)                                  │    ║
║  │  A100 / H100 — large batch training, LLM fine-tuning     │    ║
║  │  Joined as Ray worker nodes — zero code change           │    ║
║  └─────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════╝
```

### Node Configuration

```yaml
# infrastructure/node_config.yaml

head_node:
  host: 192.168.1.100        # your RTX 4060 laptop
  resources:
    GPU: 1
    CPU: 8
    memory_gb: 16
  services:
    - director_agent
    - world_state
    - blender_env
    - mutation_api_dispatcher

worker_nodes:
  - id: worker_a
    host: 192.168.1.101      # laptop 1
    resources:
      GPU: 1
      CPU: 6
    role: rl_executor

  - id: worker_b
    host: 192.168.1.102      # laptop 2
    resources:
      GPU: 1
      CPU: 6
    role: specialist_agents

  - id: worker_c
    host: 192.168.1.103      # laptop 3
    resources:
      GPU: 1
      CPU: 6
    role: training_service

cloud:
  provider: vast_ai           # or RunPod, Lambda Labs
  instance_type: A100_80GB
  autoscale: true
  trigger: training_queue_depth > 1000
```

### Joining a Node (one command)

```bash
# On head node (run once)
ray start --head --port=6379

# On every worker laptop (run once)
ray start --address=192.168.1.100:6379

# Cloud nodes join the same way via WireGuard VPN tunnel
```

---

## 4. Mutation API — Complete Action Vocabulary

```python
# mutation_api/actions.py

class ActionType(IntEnum):
    # Object Lifecycle
    SPAWN_OBJECT        = 0
    DELETE_OBJECT       = 1
    DUPLICATE_OBJECT    = 2

    # Transform (absolute)
    SET_POSITION        = 3
    SET_ROTATION        = 4
    SET_SCALE           = 5

    # Transform (relative)
    TRANSLATE_OBJECT    = 6
    ROTATE_OBJECT       = 7
    SCALE_OBJECT        = 8

    # Hierarchy
    SET_PARENT          = 9
    UNSET_PARENT        = 10
    GROUP_OBJECTS       = 11

    # Geometry
    EXTRUDE_FACE        = 12
    SUBDIVIDE_MESH      = 13
    BEVEL_EDGE          = 14
    MERGE_VERTICES      = 15
    APPLY_MODIFIER      = 16

    # Materials
    ASSIGN_MATERIAL     = 17
    SET_MATERIAL_PROP   = 18

    # Lighting
    SPAWN_LIGHT         = 19
    SET_LIGHT_PROP      = 20

    # Camera
    SPAWN_CAMERA        = 21
    SET_CAMERA_PROP     = 22

    # Scene
    SET_WORLD_PROP      = 23
    SET_PHYSICS_PROP    = 24

    # Animation
    SET_KEYFRAME        = 25
    SET_ANIM_CURVE      = 26

    # Terminal
    TASK_COMPLETE       = 27
    TASK_FAILED         = 28
```

---

## 5. Complete File & Folder Structure

```
aether/
│
├── README.md
├── pyproject.toml
├── .env.example
│
│
├── core/                               # Shared primitives — no AI logic here
│   ├── __init__.py
│   ├── types.py                        # MutationAction, SceneState, AgentTask, etc.
│   ├── message_bus.py                  # Redis Streams pub/sub abstraction
│   ├── world_state.py                  # Scene graph reads/writes (Redis + Postgres)
│   ├── registry.py                     # Object ID registry (stable int IDs)
│   └── config.py                       # Pydantic settings, loaded from .env
│
│
├── mutation_api/                       # The stable abstraction layer
│   ├── __init__.py
│   ├── api.py                          # MutationAPI class — dispatch table
│   ├── actions.py                      # ActionType enum + MutationAction dataclass
│   ├── result.py                       # MutationResult dataclass
│   ├── recorder.py                     # Logs all mutations → demonstration data
│   ├── validator.py                    # Pre-execution action validation
│   │
│   └── adapters/
│       ├── __init__.py
│       ├── base_adapter.py             # Protocol / interface all adapters implement
│       ├── blender_adapter.py          # bpy.ops translation (training backend)
│       └── engine_adapter.py           # pybind11 → C++ MutationAPI (production)
│
│
├── agents/                             # All Ray remote actor agents
│   ├── __init__.py
│   ├── base_agent.py                   # BaseAgent Ray actor + AgentTask contract
│   ├── agent_task.py                   # AgentTask / AgentTaskResult dataclasses
│   │
│   ├── director/
│   │   ├── __init__.py
│   │   ├── director_agent.py           # Ray remote actor — top-level orchestrator
│   │   ├── task_planner.py             # Intent → ordered task list (LLM call)
│   │   ├── task_monitor.py             # Tracks subtask completion via World State
│   │   └── prompts/
│   │       ├── system_prompt.txt
│   │       └── decomposition_prompt.txt
│   │
│   ├── layout/
│   │   ├── __init__.py
│   │   ├── layout_agent.py             # Spatial composition decisions
│   │   └── prompts/
│   │       └── layout_prompt.txt
│   │
│   ├── asset/
│   │   ├── __init__.py
│   │   ├── asset_agent.py              # Object type / asset selection
│   │   └── catalog.py                  # Known asset types + semantic tags
│   │
│   ├── lighting/
│   │   ├── __init__.py
│   │   └── lighting_agent.py           # Lighting design given populated scene
│   │
│   └── critic/
│       ├── __init__.py
│       ├── critic_agent.py             # Scores renders — provides reward signal
│       ├── clip_scorer.py              # CLIP-based visual quality scoring
│       ├── geometric_checker.py        # Rule-based spatial validity checks
│       └── reward_composer.py          # Combines multiple reward signals → scalar
│
│
├── executor/                           # The RL executor — core learning system
│   ├── __init__.py
│   ├── rl_executor.py                  # Ray remote actor — runs policy forward pass
│   │
│   ├── policy/
│   │   ├── __init__.py
│   │   ├── network.py                  # Full policy network (Transformer decoder)
│   │   ├── heads.py                    # ActionTypeHead, ObjectPointerHead, ParamHead
│   │   ├── observation.py              # Multi-modal observation encoder
│   │   ├── object_encoder.py           # Transformer over object list → embeddings
│   │   ├── spatial_encoder.py          # 3D CNN over voxel grid
│   │   ├── visual_encoder.py           # ViT / CLIP for rendered frame
│   │   └── fusion.py                   # Cross-attention fusion of three streams
│   │
│   ├── action_space.py                 # Gym action space definition
│   ├── observation_space.py            # Gym observation space definition
│   │
│   └── training/
│       ├── __init__.py
│       ├── imitation_trainer.py        # Phase 1 — behavior cloning (PyTorch Lightning)
│       ├── rl_trainer.py               # Phase 2 — PPO (TorchRL)
│       ├── world_model_trainer.py      # Phase 3 — Dreamer-V3
│       ├── replay_buffer.py            # Trajectory store (Redis-backed)
│       └── curriculum.py               # Task difficulty progression
│
│
├── environment/                        # Gym environment wrapping Blender
│   ├── __init__.py
│   ├── blender_env.py                  # gymnasium.Env implementation
│   ├── scene_renderer.py               # Renders observation frames via Blender
│   ├── scene_resetter.py               # Clean scene reset between episodes
│   ├── observation_builder.py          # Assembles multi-modal observation dict
│   └── task_suite/
│       ├── __init__.py
│       ├── base_task.py                # Task interface (goal, reward fn, done fn)
│       ├── spawn_task.py               # "Spawn N objects at positions"
│       ├── room_task.py                # "Build a room with walls and floor"
│       ├── placement_task.py           # "Arrange furniture in a living room"
│       └── scene_task.py               # "Build a complete outdoor scene"
│
│
├── llm/                                # LLM Gateway — provider abstraction
│   ├── __init__.py
│   ├── gateway.py                      # LLMGateway ABC + LLMRequest/Response types
│   ├── factory.py                      # create_llm_gateway() — reads LLM_MODE from env
│   └── backends/
│       ├── __init__.py
│       ├── api_backend.py              # OpenAI / Anthropic / Gemini / any API endpoint
│       └── local_backend.py            # vLLM / Ollama — local GPU inference
│
│
│   ├── __init__.py
│   ├── working_memory.py               # Redis — current task state, last N actions
│   ├── episodic_memory.py              # Postgres — session history, past mistakes
│   └── semantic_memory.py             # Long-term knowledge (domain facts)
│
│
├── data/                               # Demonstration data pipeline
│   ├── __init__.py
│   ├── collector.py                    # Records human Blender sessions
│   ├── processor.py                    # Raw log → training-ready trajectory
│   ├── augmentor.py                    # Trajectory augmentation
│   ├── procedural_generator.py         # Synthetic demonstration generation
│   └── schema.py                       # Trajectory / demonstration dataclasses
│
│
├── serving/                            # Production inference serving
│   ├── __init__.py
│   ├── policy_server.py                # Serves trained policy (TorchScript/ONNX)
│   └── llm_server.py                   # vLLM server wrapper for agents
│
│
├── infrastructure/
│   ├── node_config.yaml                # Per-node resource declarations
│   ├── ray_cluster.py                  # Cluster init, health checks, node mgmt
│   ├── docker-compose.yml              # Redis + Postgres + monitoring stack
│   ├── Dockerfile.worker               # Worker node container
│   └── wireguard/
│       └── wg0.conf.example            # VPN config for LAN + cloud mesh
│
│
├── blender_entry/                      # Blender-side entry point (addon)
│   ├── __init__.py                     # Blender addon registration
│   ├── blender_manifest.toml           # Blender 4.x manifest
│   ├── aether_panel.py                 # Sidebar UI panel in Blender
│   ├── aether_operators.py             # Blender operators (start/stop/connect)
│   ├── session_bridge.py               # WebSocket client → Aether head node
│   ├── action_executor.py              # Receives actions, calls bpy on main thread
│   └── observation_exporter.py         # Exports scene state → head node
│
│
└── tools/
    ├── benchmark.py                    # Policy benchmark suite
    ├── visualize_trajectory.py         # Replay recorded trajectories
    └── export_policy.py                # Export policy → TorchScript / ONNX
```

---

## 6. Blender Entry Point — How It Connects

Blender runs as a **slave environment** during training. The Aether head node controls it entirely. Here's the connection architecture:

```
HEAD NODE (Python/Ray)                    BLENDER PROCESS
                                          
BlenderEnv.step(action)                   
    │                                     
    │  WebSocket (localhost or LAN)        
    ▼                                     
session_bridge.py  ──────────────────▶   action_executor.py
(client)                                  (server, main thread)
                                              │
                                              ▼
                                          blender_adapter.py
                                              │
                                              ▼
                                          bpy.ops / bpy.data
                                              │
                                              ▼
                   ◀──────────────────   observation_exporter.py
                   SceneState + render
```

### Blender Addon Entry (`blender_entry/__init__.py`)

```python
bl_info = {
    "name": "Aether",
    "author": "Siddhant",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "category": "3D View",
    "description": "Aether AI training environment bridge",
}

import bpy
from .aether_operators import (
    AETHER_OT_StartSession,
    AETHER_OT_StopSession,
)
from .aether_panel import AETHER_PT_Panel
from .session_bridge import AetherBridge

_bridge: AetherBridge | None = None

def register():
    bpy.utils.register_class(AETHER_OT_StartSession)
    bpy.utils.register_class(AETHER_OT_StopSession)
    bpy.utils.register_class(AETHER_PT_Panel)

def unregister():
    bpy.utils.unregister_class(AETHER_PT_Panel)
    bpy.utils.unregister_class(AETHER_OT_StopSession)
    bpy.utils.unregister_class(AETHER_OT_StartSession)
```

### Session Bridge (`blender_entry/session_bridge.py`)

```python
import asyncio
import json
import threading
import websockets
from .action_executor import ActionExecutor
from .observation_exporter import ObservationExporter

class AetherBridge:
    """
    WebSocket client running in a background thread.
    Receives MutationActions from head node.
    Dispatches to ActionExecutor on Blender main thread.
    Sends SceneState + render back.
    """

    def __init__(self, host: str, port: int = 8765):
        self.uri = f"ws://{host}:{port}/blender"
        self.executor = ActionExecutor()
        self.exporter = ObservationExporter()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True
        )

    def start(self):
        self._thread.start()

    def _run_loop(self):
        asyncio.run(self._connect())

    async def _connect(self):
        async with websockets.connect(self.uri) as ws:
            async for message in ws:
                action = json.loads(message)
                # Execute on Blender main thread via bpy.app.timers
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.executor.execute, action
                )
                obs = self.exporter.export()
                await ws.send(json.dumps({
                    "result": result,
                    "observation": obs,
                }))
```

---

## 7. LLM Gateway — Provider Abstraction

Every agent calls this. No agent ever imports `openai` or `transformers` directly.

### Interface

```python
# llm/gateway.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMRequest:
    system_prompt: str
    user_prompt: str
    max_tokens: int = 1024
    temperature: float = 0.3
    cache_key: str | None = None     # set this to enable Redis response cache

@dataclass
class LLMResponse:
    content: str
    provider: str                    # "openai" | "local" | etc.
    cached: bool
    latency_ms: float

class LLMGateway(ABC):
    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse: ...
```

### API Backend

```python
# llm/backends/api_backend.py

import openai
import time
from llm.gateway import LLMGateway, LLMRequest, LLMResponse
from core.cache import ResponseCache

class APIBackend(LLMGateway):
    """
    Supports any OpenAI-compatible API endpoint.
    OpenAI, Anthropic (via proxy), Gemini, Together, etc.
    Switch model/base_url in config — zero code changes.
    """

    def __init__(self, cache: ResponseCache):
        self.client = openai.AsyncOpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.api_base_url,   # swap endpoint here
        )
        self.model = settings.llm.api_model        # "gpt-4o", "gpt-4o-mini", etc.
        self.cache = cache

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if request.cache_key:
            cached = await self.cache.get(request.cache_key)
            if cached:
                return LLMResponse(
                    content=cached, provider=self.model,
                    cached=True, latency_ms=0
                )

        t0 = time.monotonic()
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user",   "content": request.user_prompt},
            ],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        latency = (time.monotonic() - t0) * 1000
        content = response.choices[0].message.content

        if request.cache_key:
            await self.cache.set(request.cache_key, content, ttl=3600)

        return LLMResponse(
            content=content, provider=self.model,
            cached=False, latency_ms=latency
        )
```

### Local Backend

```python
# llm/backends/local_backend.py

import httpx
from llm.gateway import LLMGateway, LLMRequest, LLMResponse

class LocalBackend(LLMGateway):
    """
    Talks to a local vLLM or Ollama server.
    OpenAI-compatible endpoint — same interface as API backend.
    """

    def __init__(self, cache: ResponseCache):
        self.base_url = settings.llm.local_base_url   # e.g. http://localhost:8000/v1
        self.model = settings.llm.local_model          # "llama3.1:8b", "qwen2.5:7b"
        self.cache = cache
        self.client = openai.AsyncOpenAI(
            api_key="local",
            base_url=self.base_url,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        # identical logic to APIBackend — same OpenAI client, different endpoint
        ...
```

### Gateway Factory

```python
# llm/factory.py

from core.config import settings
from llm.backends.api_backend import APIBackend
from llm.backends.local_backend import LocalBackend
from core.cache import ResponseCache

def create_llm_gateway(cache: ResponseCache) -> LLMGateway:
    """
    mode = "api"   → ChatGPT / Anthropic / Gemini / any API
    mode = "local" → vLLM / Ollama on local GPU
    
    Change LLM_MODE in .env — nothing else changes.
    """
    if settings.llm.mode == "api":
        return APIBackend(cache)
    elif settings.llm.mode == "local":
        return LocalBackend(cache)
    else:
        raise ValueError(f"Unknown LLM mode: {settings.llm.mode}")
```

### Config (`.env`)

```bash
# Switch between API and local with one line

LLM_MODE=api                          # "api" or "local"

# API backend (when LLM_MODE=api)
LLM_API_KEY=sk-...                    # your ChatGPT / Anthropic key
LLM_API_BASE_URL=https://api.openai.com/v1
LLM_API_MODEL=gpt-4o

# Local backend (when LLM_MODE=local)
LLM_LOCAL_BASE_URL=http://localhost:8000/v1
LLM_LOCAL_MODEL=llama3.1:8b
```

### How Agents Use It

```python
# agents/director/director_agent.py  (example)

@ray.remote
class DirectorAgent(BaseAgent):

    def __init__(self):
        cache = ResponseCache()
        self.llm = create_llm_gateway(cache)   # ← only line that matters

    async def decompose_intent(self, intent: str) -> TaskPlan:
        response = await self.llm.complete(LLMRequest(
            system_prompt=DIRECTOR_SYSTEM_PROMPT,
            user_prompt=intent,
            cache_key=f"plan:{hash(intent)}",  # same intent = free
        ))
        return TaskPlan.parse(response.content)

    # Director never knows if it's talking to GPT-4o or Llama.
    # It never will.
```

---

## 8. Core Contracts (Never Break These)

### MutationAdapter Protocol
```python
# mutation_api/adapters/base_adapter.py

from typing import Protocol
from core.types import MutationAction, MutationResult, SceneState

class MutationAdapter(Protocol):
    def execute(self, action: MutationAction) -> MutationResult: ...
    def get_scene_state(self) -> SceneState: ...
    def reset(self) -> SceneState: ...
    def render_frame(self) -> bytes: ...          # PNG bytes
```

### BaseAgent Contract
```python
# agents/base_agent.py

import ray
from core.types import AgentTask, AgentTaskResult, AgentStatus

class BaseAgent:
    """All specialist agents inherit this. Ray remote decorator applied by subclass."""

    def assign_task(self, task: AgentTask) -> AgentTaskResult:
        raise NotImplementedError

    def get_status(self) -> AgentStatus:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError
```

### WorldState Contract
```python
# core/world_state.py

from typing import Protocol
from core.types import SceneGraph, MutationResult, Task

class WorldStateProtocol(Protocol):
    def get_scene(self) -> SceneGraph: ...
    def apply_mutation(self, result: MutationResult) -> None: ...
    def push_task(self, task: Task) -> None: ...
    def pop_task(self) -> Task | None: ...
    def get_history(self, n: int) -> list[MutationResult]: ...
```

---

## 8. Technology Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| Orchestration | **Ray 2.x** | Distributed actors, cluster mgmt, zero topology coupling |
| Message bus | **Redis Streams** | Decoupled pub/sub, durable, fast |
| Hot state | **Redis** | Sub-ms scene graph reads + LLM response cache |
| Persistent state | **Postgres** | Trajectories, episodic memory, history |
| Policy network | **PyTorch 2.x** | Full control over architecture |
| BC training | **PyTorch Lightning** | Clean training loops |
| RL training | **TorchRL** | PPO, SAC, replay buffers |
| World model | **Dreamer-V3** | Imagination-based planning |
| **LLM Gateway** | **Custom abstraction** | Single interface — agents never call providers directly |
| LLM API backend | **OpenAI SDK** (async) | GPT-4o / GPT-4o-mini — set via `.env`, zero code change |
| LLM local backend | **vLLM + Ollama** | Llama 3.1 8B, Qwen 2.5, DeepSeek — runs on local GPU |
| LLM switching | **`LLM_MODE=api\|local`** | One env var — architecture stays identical |
| Visual encoder | **CLIP (ViT-L/14)** | Semantic visual understanding |
| Engine bridge | **pybind11** | Zero-overhead C++ ↔ Python |
| Blender bridge | **WebSocket** | Clean process boundary |
| Experiment tracking | **Weights & Biases** | All training runs, all nodes |
| VPN mesh | **WireGuard** | LAN + cloud node networking |
| Config | **Pydantic Settings** | Typed, env-driven, validated |

---

## 9. Build Order

```
Phase 0 — Foundation (start here)
├── core/types.py                   Define all shared dataclasses
├── llm/gateway.py                  LLM abstraction (before any agent)
├── llm/backends/api_backend.py     ChatGPT backend — usable immediately
├── llm/backends/local_backend.py   Local backend — plug in when ready
├── llm/factory.py                  create_llm_gateway() factory
├── mutation_api/actions.py         Full action vocabulary
├── mutation_api/adapters/base      Adapter protocol
├── mutation_api/adapters/blender   Blender bpy translation
└── blender_entry/                  Addon + WebSocket bridge

Phase 1 — Environment + Data
├── environment/blender_env.py      Gym environment
├── environment/task_suite/         First 2 tasks (spawn, room)
├── data/collector.py               Demonstration recorder
└── data/processor.py               Log → trajectory

Phase 2 — Executor (Learning)
├── executor/policy/                Full policy network
├── executor/training/imitation     BC trainer
└── executor/training/rl_trainer    PPO trainer

Phase 3 — Agents
├── agents/director/                Director Agent (Ray actor)
├── agents/layout/                  Layout Agent
└── agents/critic/                  Critic + reward signal

Phase 4 — Distribution
├── infrastructure/ray_cluster      Multi-node Ray setup
├── infrastructure/docker-compose   Redis + Postgres
└── infrastructure/wireguard        LAN + cloud VPN mesh

Phase 5 — Scale + World Model
├── executor/training/world_model   Dreamer-V3
├── agents/lighting/                Lighting Agent
└── serving/policy_server           Production policy serving
```

---

## 10. Scaling Path (Zero Code Changes)

```
TODAY                    NEAR TERM               CLOUD
─────────────────────    ─────────────────────   ──────────────────
Ray: 1 node              Ray: 4 LAN nodes        Ray: 4 local + N cloud
RTX 4060                 + 3 laptop GPUs         + A100s on demand

All services             Services distributed    Training bursts
on one GPU               by role                 to cloud auto-scaled

ray start --head         ray start --address=    Same command
                         <head>:6379             via WireGuard
```

The AI system code is identical across all three phases.
Only `infrastructure/node_config.yaml` changes.

---

*Aether — designed to scale from a single laptop to a distributed cloud cluster without touching AI logic. LLM provider is a config choice, not an architectural one.*
