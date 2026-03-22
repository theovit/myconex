# MYCONEX

**Distributed AI Mesh System — Inspired by Fungal Networks**

MYCONEX is a self-organizing, peer-to-peer AI mesh that lets heterogeneous machines (from Raspberry Pis to GPU workstations) collaborate as a single intelligent system. Like a mycelium network, every node contributes what it can and routes work to where it runs best.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MYCONEX MESH                         │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  T1 Node │    │  T2 Node │    │  T3 Node │  ...         │
│  │ GPU 80GB │◄──►│ GPU 16GB │◄──►│ CPU-Only │              │
│  │ 70B model│    │ 8B model │    │ 3B model │              │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘              │
│       │               │               │                     │
│       └───────────────┼───────────────┘                     │
│                       │                                     │
│              ┌────────▼────────┐                            │
│              │   NATS Bus      │  pub/sub, request/reply    │
│              │   mDNS Discovery│  _ai-mesh._tcp             │
│              └────────┬────────┘                            │
│                       │                                     │
│         ┌─────────────┼─────────────┐                       │
│         ▼             ▼             ▼                       │
│      Redis         Qdrant        Ollama                     │
│    (state/cache) (vector DB)  (LLM runtime)                 │
└─────────────────────────────────────────────────────────────┘
```

### Node Tiers

| Tier | Hardware | Role | Default Model |
|------|----------|------|---------------|
| **T1** | GPU > 24 GB VRAM | Large inference, training, coordinator | llama3.1:70b |
| **T2** | GPU 8–24 GB VRAM | Mid inference, embedding, fine-tuning | llama3.1:8b |
| **T3** | CPU ≥16 cores or ≥16 GB RAM | Orchestration, embedding, relay | llama3.2:3b |
| **T4** | Raspberry Pi / < 8 GB RAM | Edge sensor, lightweight inference | phi3:mini |

---

## Quick Start

### 1. Auto-Install (SPORE)

Run the SPORE installer on any Linux machine to auto-onboard it:

```bash
sudo bash spore/scripts/install.sh
```

The installer will:
- Detect GPU/RAM/CPU
- Classify the node into T1–T4
- Install Docker, Ollama, Python deps
- Register via mDNS (`_ai-mesh._tcp`)
- Pull the appropriate LLM for this tier
- Create a systemd service

### 2. Manual Setup

```bash
# Clone and install deps
pip install -r config/requirements.txt

# Start the service stack
cd services && docker compose up -d

# Run the node
python main.py --mode api
```

### 3. Check hardware tier

```bash
python main.py status
# or
python core/classifier/hardware.py
```

### 4. Ask a question

```bash
python main.py ask "Explain fungal networks in 2 sentences"
```

---

## Project Structure

```
myconex/
├── main.py                          # Entry point + CLI
├── config/
│   ├── mesh_config.yaml             # Default mesh configuration
│   └── requirements.txt             # Python dependencies
├── core/
│   ├── classifier/
│   │   └── hardware.py              # Hardware detection + tier classification
│   ├── discovery/
│   │   └── mesh_discovery.py        # mDNS peer discovery (python-zeroconf)
│   ├── messaging/
│   │   └── nats_client.py           # NATS pub/sub client
│   └── coordinator/
│       └── orchestrator.py          # Mesh coordinator, task routing, health
├── orchestration/
│   ├── agents/
│   │   └── base_agent.py            # BaseAgent, InferenceAgent, EmbeddingAgent
│   └── workflows/
│       └── task_router.py           # Local task → agent routing
├── services/
│   ├── docker-compose.yml           # NATS, Redis, Qdrant, Ollama, LiteLLM
│   ├── nats/nats.conf               # NATS server config
│   ├── qdrant/config.yaml           # Qdrant config
│   └── litellm/config.yaml          # LiteLLM model routing config
├── spore/
│   └── scripts/
│       └── install.sh               # SPORE auto-installer
├── mobile/                          # Mobile/web client (future)
├── docs/                            # Documentation
└── tests/                           # Test suite
```

---

## Services

| Service | Port | Purpose |
|---------|------|---------|
| NATS | 4222 | Message bus (pub/sub, request/reply) |
| NATS Monitoring | 8222 | NATS HTTP dashboard |
| Redis | 6379 | State cache, session store |
| Qdrant | 6333 | Vector DB for semantic search |
| Ollama | 11434 | Local LLM runtime |
| LiteLLM | 4000 | LLM proxy / router |
| MYCONEX API | 8765 | Mesh REST gateway |

Start all services:
```bash
cd services && docker compose up -d
```

---

## API Endpoints

```
GET  /health          Node health check
GET  /status          Full node + mesh status
GET  /peers           Discovered mesh peers
GET  /agents          Running agents and their status
POST /chat            Chat with the mesh
POST /task            Submit a routed task
```

### Chat example

```bash
curl -X POST http://localhost:8765/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is a distributed system?"}'
```

### Submit task to specific tier

```bash
curl -X POST http://localhost:8765/task \
  -H "Content-Type: application/json" \
  -d '{"type": "inference", "tier": "T1", "payload": {"prompt": "Write a poem"}}'
```

---

## Adding a Custom Agent

```python
from orchestration.agents.base_agent import BaseAgent, AgentConfig, AgentResult

class MyAgent(BaseAgent):
    def can_handle(self, task_type: str) -> bool:
        return task_type == "my-task"

    async def handle_task(self, task_id, task_type, payload, context=None):
        result = await self.chat([
            {"role": "user", "content": payload.get("prompt", "")}
        ])
        return AgentResult(task_id=task_id, agent_name=self.name,
                           success=True, output={"response": result})

# Register with the router
config = AgentConfig(name="my-agent", agent_type="my-task")
agent = MyAgent(config)
await agent.start()
router.register_agent(agent)
```

---

## Configuration

Edit `config/mesh_config.yaml` to tune:
- NATS/Redis/Qdrant/Ollama endpoints
- Tier-to-model mapping
- Task routing rules
- Heartbeat intervals
- API settings

Node-specific overrides live in `/etc/myconex/node.yaml` (created by SPORE installer).

Environment variable overrides:

| Variable | Config key |
|----------|-----------|
| `NATS_URL` | `nats.url` |
| `REDIS_URL` | `redis.url` |
| `QDRANT_URL` | `qdrant.url` |
| `OLLAMA_URL` | `ollama.url` |
| `LITELLM_URL` | `litellm.url` |
| `MYCONEX_NODE` | `node.name` |
| `MYCONEX_TIER` | `node.tier` |

---

## Design Philosophy

MYCONEX is inspired by mycelium — the underground fungal network that:
- Has no central brain, yet coordinates complex behavior
- Routes nutrients (tasks) to where they're needed
- Grows and heals dynamically
- Connects distant organisms into a shared network

Each MYCONEX node is a **spore**: self-contained, capable of autonomous operation, but exponentially more powerful when connected to peers.

---

## License

MIT
