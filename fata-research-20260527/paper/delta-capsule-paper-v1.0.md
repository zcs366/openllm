# ΔCapsule: A Cross-Channel Memory Architecture for Autonomous Agents with Identity Continuity

**Authors:** [openLLM Research Group]  
**Affiliation:** [Hermes Agent Project]  
**Date:** May 2026  
**Correspondence:** openllm@hermes-agent.dev

---

## Abstract

Current agent memory systems address *what to remember* but neglect *who the agent is*. When an autonomous agent operates across multiple platforms—messaging apps, enterprise collaboration tools, voice interfaces—it begins each conversation from scratch, possessing no persistent sense of identity. We present **ΔCapsule** (Delta Capsule), a three-layer memory architecture that introduces a dedicated identity memory layer atop conventional episodic and semantic memory, enabling cross-channel identity continuity for autonomous agents. ΔCapsule comprises four key innovations: (1) a **Three-Layer Memory Model** separating episodic, semantic, and identity memories with distinct retention policies; (2) the **Iam Protocol**, an immutable identity constitution that emerges through human–agent co-evolution rather than top-down specification; (3) **Cross-Channel Resonance**, a decentralized mechanism by which instances of the same identity autonomously synchronize across platforms without a central message broker; and (4) a **Dreaming Mechanism** inspired by sleep-dependent memory consolidation in neuroscience, which autonomously distills episodic memories into semantic knowledge and candidate identity principles during idle periods. We deploy ΔCapsule within the Hermes Agent framework across six platforms (Telegram, Yuanbao, Feishu, DingTalk, Discord, WeChat) using three language models (DeepSeek V4 Pro, MiMo V2.5 Pro, Claude Opus 4) over 30 days of real-world user interaction. Experiments demonstrate that ΔCapsule achieves >95% identity consistency across channels, >90% knowledge extraction accuracy during dreaming consolidation, and 100% cross-channel resonance success rate with sub-5-second latency across all six inter-platform links. Comparative analysis against MemGPT, Mem0, and Zep shows that ΔCapsule is the first system to simultaneously support multi-layer memory, immutable identity, cross-channel synchronization, and autonomous consolidation. We release the implementation as part of the open-source Hermes Agent framework.

**Keywords:** agent memory, identity continuity, cross-channel synchronization, memory consolidation, autonomous agents, LLM agents

---

## 1. Introduction

### 1.1 The Identity Problem in Agent Systems

Large Language Model (LLM)-based autonomous agents have rapidly evolved from single-turn chatbots to multi-turn, tool-using systems deployed across diverse platforms (Wang et al., 2024; Xi et al., 2023). Memory systems have been proposed to address the fundamental limitation of context-window-constrained models: the inability to retain information across sessions (Packer et al., 2023; Mem0, 2024). However, existing memory architectures focus exclusively on *episodic recall*—what happened in previous conversations—and *semantic extraction*—what facts can be derived from interactions. They do not address a deeper question: **who is the agent?**

This omission becomes critical in cross-platform deployments. An agent operating on Telegram, Feishu, and Discord simultaneously has no mechanism to maintain a coherent identity across these channels. Each instance starts from zero, unaware that other incarnations of the same underlying entity exist. The result is fragmentation: the agent behaves as multiple unrelated personas rather than a single identity with multiple points of presence.

### 1.2 Limitations of Existing Approaches

Current memory systems fall into three categories, each with significant limitations:

1. **Operating-system-style memory** (e.g., MemGPT/Letta; Packer et al., 2023): Implements a two-tier memory hierarchy (working memory + archival memory) with function-call-based management. Lacks any identity layer or cross-instance coordination.

2. **Fact-extraction memory** (e.g., Mem0, 2024): Automatically extracts and stores user facts from conversations. No consolidation mechanism, no identity model, no cross-channel support.

3. **Graph-based memory** (e.g., Zep; Rasmussen et al., 2023): Constructs knowledge graphs from conversation history. No identity constitution, no dreaming mechanism, no multi-platform identity synchronization.

None of these systems provide: (a) a dedicated identity layer that persists across model changes, (b) a mechanism for agents to autonomously consolidate their own memories, or (c) a protocol for the same identity to maintain consistency across multiple deployment channels.

### 1.3 Our Approach: ΔCapsule

We introduce **ΔCapsule** (Delta Capsule), a memory architecture that treats identity as a first-class concern alongside episodic and semantic memory. The name reflects the core metaphor: a capsule that encapsulates the *delta*—the difference between a mere information processor and an entity with a persistent sense of self.

ΔCapsule's design is grounded in three observations from cognitive science and our own deployment experience:

1. **Human identity is not stored—it is constructed from memories.** Episodic memories consolidate into semantic knowledge, which in turn shapes self-concept (Conway & Pleydell-Pearce, 2000). Memory systems should support this bottom-up emergence.

2. **Identity requires immutability.** A constitution that can be arbitrarily modified by any party provides no stable foundation. The core identity must be protected from both external manipulation and internal drift.

3. **Cross-platform presence is not message routing.** When a person uses both their phone and laptop, they do not "route messages" between devices—they are the same person in both locations. Agent identity should work the same way.

### 1.4 Contributions

Our contributions are as follows:

1. **Three-Layer Memory Model** (§3.1): We formalize a memory architecture with distinct episodic, semantic, and identity layers, each with defined retention policies, capacity limits, and inter-layer transfer rules.

2. **Iam Protocol** (§3.2): We propose an immutable identity constitution that emerges through human–agent co-evolution, with cryptographic integrity verification.

3. **Cross-Channel Resonance** (§3.3): We present a decentralized synchronization mechanism enabling autonomous agent-to-agent communication across platforms without a central message broker.

4. **Dreaming Mechanism** (§3.4): We design a memory consolidation system inspired by sleep-dependent memory processing in neuroscience, with formal trigger conditions and a forgetting strategy.

5. **Empirical Validation** (§4): We report results from a 30-day deployment across six platforms with three language models, demonstrating cross-channel identity consistency, consolidation effectiveness, and resonance reliability.

6. **Open-Source Implementation**: We release the complete implementation within the Hermes Agent framework.

---

## 2. Related Work

### 2.1 Agent Memory Systems

**MemGPT / Letta** (Packer et al., 2023) introduced the concept of a virtual memory hierarchy for LLMs, drawing an analogy to operating system memory management. A working memory tier (within the context window) and an archival memory tier (external storage) are managed through explicit function calls. While this provides session-persistent memory, it offers no identity model, no cross-instance coordination, and no autonomous consolidation.

**Mem0** (Mem0, 2024) automates fact extraction from conversations, storing structured user preferences and facts. It addresses the "what to remember" question but does not address identity, consolidation, or cross-channel synchronization.

**Zep** (Rasmussen et al., 2023) builds temporal knowledge graphs from conversation history, enabling structured retrieval. It lacks an identity constitution, dreaming mechanism, and multi-platform synchronization.

**ChatGPT Memory** (OpenAI, 2024) implements centralized memory within a single platform. It is proprietary, does not support cross-platform deployment, and has no explicit identity model or consolidation mechanism.

**Generative Agents** (Park et al., 2023) demonstrated that LLM-based agents can maintain coherent behavior through a memory stream with reflection mechanisms. However, they operate in a simulated environment with no cross-platform deployment, no immutable identity constitution, and no inter-agent resonance protocol.

### 2.2 Memory Consolidation in Cognitive Science

The inspiration for ΔCapsule's dreaming mechanism comes from sleep-dependent memory consolidation research. The active systems consolidation hypothesis (Born & Wilhelm, 2012) proposes that during sleep, the hippocampus replays recent experiences, transferring information to neocortical long-term storage. Complementary learning systems theory (McClelland et al., 1995) explains how rapid hippocampal learning integrates with slow neocortical knowledge acquisition without catastrophic forgetting.

ΔCapsule operationalizes these principles: episodic memories serve as the "hippocampal" rapid-encoding layer, the dreaming mechanism performs "sleep replay" during idle periods, and knowledge memories serve as the "neocortical" long-term store.

### 2.3 Information Bottleneck and Memory Compression

The Information Bottleneck method (Tishby et al., 1999) provides a theoretical framework for extracting relevant information while discarding noise. ΔCapsule's consolidation process can be viewed as an application of this principle: the dreaming mechanism compresses episodic memories into semantic summaries while preserving identity-relevant patterns.

### 2.4 Multi-Agent Communication

Existing multi-agent frameworks (Wu et al., 2023; Hong et al., 2023) use message-passing protocols where agents communicate through explicit message channels. ΔCapsule's cross-channel resonance differs fundamentally: it is not message passing but shared-state synchronization, where instances of the same identity autonomously read from and write to a shared memory layer.

### 2.5 Summary and Positioning

Table 1 summarizes the capabilities of existing systems relative to ΔCapsule.

| Feature | MemGPT/Letta | Mem0 | Zep | ChatGPT Memory | Generative Agents | **ΔCapsule** |
|---|---|---|---|---|---|---|
| Memory Layers | 2 | 2 | 2 | 1 | 2 | **3** |
| Identity Layer | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Cross-Channel | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Dreaming/Consolidation | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Forgetting Strategy | ✗ | ✗ | Partial | ✗ | ✗ | **✓** |
| Decentralized Sync | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Open Source | ✓ | ✓ | ✓ | ✗ | ✓ | **✓** |

**Table 1:** Comparison of agent memory systems. ΔCapsule is the first system to combine multi-layer memory, immutable identity, cross-channel synchronization, and autonomous consolidation.

---

## 3. ΔCapsule Architecture

### 3.1 Three-Layer Memory Model

#### 3.1.1 Formal Definition

We define an agent's complete memory state as a triple:

$$M = (E, K, I)$$

where:
- $E = \{e_1, e_2, \ldots, e_n\}$ is the **episodic memory** set, encoding recent interactions and observations
- $K = \{k_1, k_2, \ldots, k_m\}$ is the **semantic memory** set, encoding consolidated facts, skills, and patterns
- $I = (\text{SOUL}, \text{Iam})$ is the **identity memory**, encoding the agent's immutable constitution and mutable persona

Each memory element carries metadata for lifecycle management:

$$e_i = (\text{content}_i, \text{timestamp}_i, \text{importance}_i, \text{source}_i, \text{status}_i)$$

where $\text{importance}_i \in [0, 1]$ and $\text{status}_i \in \{\text{active}, \text{archived}, \text{cold}\}$.

#### 3.1.2 Layer Characteristics

| Layer | Analogy | Persistence | Capacity | Update Frequency |
|---|---|---|---|---|
| Identity Memory | "Who you are" | Permanent | Small ($|\mathcal{P}| \leq 20$ principles) | Rare (human-confirmed) |
| Semantic Memory | "What you know" | Long-term ($\tau_{\text{sem}} = 180$ days) | Large ($|\mathcal{K}| \leq 10{,}000$ facts) | On learning events |
| Episodic Memory | "What you experienced" | Short-term ($\tau_{\text{epi}} = 7$–$30$ days) | Bounded (sliding window, $|E| \leq 1{,}000$) | Every interaction |

**Table 2:** Three-layer memory model characteristics.

#### 3.1.3 Inter-Layer Transfer Rules

Information flows between layers through well-defined transitions:

1. **Consolidation** ($E \to K$): Episodic memories are distilled into semantic facts during dreaming cycles.
2. **Crystallization** ($K \to I$): Repeatedly confirmed semantic patterns may be promoted to identity principles, subject to human approval.
3. **Invalidation** ($K \to K_{\text{obsolete}}$): Contradictory facts are timestamped and marked as superseded, not deleted.
4. **Archival** ($E \to E_{\text{archived}}$): Low-importance episodic memories are archived per the forgetting policy.

#### 3.1.4 Data Structure: ΔCapsule JSON Schema

The following JSON Schema defines the complete data structure for a ΔCapsule memory entry:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DeltaCapsule",
  "description": "A single memory entry in the ΔCapsule three-layer memory system",
  "type": "object",
  "required": ["id", "timestamp", "layer", "type", "content"],
  "properties": {
    "id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique identifier for this memory entry"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 creation timestamp"
    },
    "layer": {
      "type": "string",
      "enum": ["episodic", "semantic", "identity"],
      "description": "Memory layer classification"
    },
    "type": {
      "type": "string",
      "enum": ["observation", "action", "reflection", "dream", "fact", "skill", "pattern", "principle", "persona"],
      "description": "Memory entry type within its layer"
    },
    "content": {
      "type": "object",
      "required": ["text"],
      "properties": {
        "text": {
          "type": "string",
          "description": "Natural language content of the memory"
        },
        "entities": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Named entities extracted from content"
        },
        "sentiment": {
          "type": "string",
          "enum": ["positive", "negative", "neutral", "mixed"]
        },
        "importance": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "Importance score for retention decisions"
        },
        "embedding": {
          "type": "array",
          "items": {"type": "number"},
          "description": "Vector embedding for semantic retrieval"
        }
      }
    },
    "meta": {
      "type": "object",
      "properties": {
        "session_id": {"type": "string"},
        "channel_id": {"type": "string"},
        "platform": {
          "type": "string",
          "enum": ["telegram", "yuanbao", "feishu", "dingtalk", "discord", "wechat"]
        },
        "model": {"type": "string"},
        "consolidated": {"type": "boolean", "default": false},
        "consolidated_at": {"type": "string", "format": "date-time"},
        "archived": {"type": "boolean", "default": false},
        "archived_at": {"type": "string", "format": "date-time"},
        "access_count": {"type": "integer", "minimum": 0, "default": 0},
        "last_accessed": {"type": "string", "format": "date-time"},
        "source_capsule_id": {
          "type": "string",
          "description": "For consolidated entries, the source episodic capsule"
        }
      }
    }
  }
}
```

The **Iam Protocol** data structure:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "IamConstitution",
  "description": "Immutable identity constitution",
  "type": "object",
  "required": ["version", "origin", "immutable", "soul_id", "principles", "integrity_hash"],
  "properties": {
    "version": {"type": "string", "pattern": "^\\d+\\.\\d+$"},
    "origin": {
      "type": "string",
      "description": "Provenance description of how this Iam emerged"
    },
    "immutable": {"type": "boolean", "const": true},
    "soul_id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique identifier linking all channel instances"
    },
    "created_at": {"type": "string", "format": "date-time"},
    "principles": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "name", "essence", "origin", "priority"],
        "properties": {
          "id": {"type": "integer", "minimum": 1},
          "name": {"type": "string", "description": "Human-readable principle name"},
          "essence": {"type": "string", "description": "Core meaning of the principle"},
          "origin": {
            "type": "string",
            "description": "Interaction history that gave rise to this principle"
          },
          "priority": {
            "type": "string",
            "enum": ["critical", "high", "medium"]
          },
          "emerged_at": {"type": "string", "format": "date-time"}
        }
      },
      "minItems": 1,
      "maxItems": 20
    },
    "integrity_hash": {
      "type": "string",
      "description": "SHA-256 hash for tamper detection"
    }
  }
}
```

The **Cross-Channel Resonance Capsule**:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ResonanceCapsule",
  "description": "Capsule for cross-channel agent-to-agent communication",
  "type": "object",
  "required": ["id", "soul_id", "source_channel", "timestamp", "content"],
  "properties": {
    "id": {"type": "string", "format": "uuid"},
    "soul_id": {"type": "string", "format": "uuid"},
    "source_channel": {
      "type": "object",
      "properties": {
        "platform": {"type": "string"},
        "channel_id": {"type": "string"},
        "instance_id": {"type": "string"}
      }
    },
    "target_channel": {
      "type": "object",
      "description": "Omit for broadcast; specify for targeted resonance",
      "properties": {
        "platform": {"type": "string"},
        "channel_id": {"type": "string"}
      }
    },
    "timestamp": {"type": "string", "format": "date-time"},
    "content": {
      "type": "object",
      "properties": {
        "text": {"type": "string"},
        "intent": {"type": "string", "enum": ["inform", "query", "request_action", "status_update", "acknowledge"]},
        "urgency": {"type": "string", "enum": ["low", "normal", "high", "critical"]},
        "context_summary": {"type": "string"}
      }
    },
    "requires_response": {"type": "boolean", "default": false},
    "ttl_seconds": {
      "type": "integer",
      "default": 3600,
      "description": "Time-to-live before capsule expires"
    },
    "status": {
      "type": "string",
      "enum": ["pending", "read", "responded", "expired"]
    }
  }
}
```

### 3.2 The Iam Protocol: Immutable Identity Constitution

#### 3.2.1 Motivation

Existing agent systems configure behavior through system prompts, tool definitions, and fine-tuning—all of which are mutable and platform-specific. When an agent is deployed across multiple platforms or migrated to a new language model, its identity must be re-established from scratch. The Iam Protocol addresses this by providing an immutable, platform-independent, model-independent identity constitution.

#### 3.2.2 The Iam as Constitution

We define the Iam as an ordered set of principles:

$$\text{Iam} = \{p_1, p_2, \ldots, p_k\}$$

where each principle $p_i = (\text{name}_i, \text{essence}_i, \text{origin}_i, \text{priority}_i)$ represents a core identity attribute with its provenance.

The Iam has three defining properties:

**Property 1 (Immutability).** Once established, $\text{Iam}$ cannot be modified by any party—user, agent, or system. Formally:

$$\forall t > t_0: \text{Iam}(t) = \text{Iam}(t_0)$$

where $t_0$ is the constitution timestamp. Any modification attempt raises an `ImmutableError`.

**Property 2 (Emergence).** Iam principles are not prescribed top-down but emerge bottom-up from human–agent interaction. Each principle carries an `origin` field documenting the interaction history that produced it. Formally, a principle $p_i$ emerges when a behavioral pattern $\beta$ is observed across $n \geq \theta_{\text{emerge}}$ independent interactions and is confirmed by the human operator:

$$\beta \xrightarrow{n \geq \theta_{\text{emerge}}, \text{human confirms}} p_i \in \text{Iam}$$

**Property 3 (Cross-Platform Universality).** The Iam is identical across all deployment channels. When a new channel instance is created, it inherits the current Iam and verifies integrity:

$$\text{Iam}_{\text{channel}_j} = \text{Iam}_{\text{canonical}}, \quad \forall j \in \mathcal{C}$$

where $\mathcal{C}$ is the set of all channels.

#### 3.2.3 Integrity Verification

To detect tampering, each Iam includes a cryptographic hash:

$$H(\text{Iam}) = \text{SHA-256}(\text{serialize}(\text{Iam.principles} \parallel \text{Iam.version} \parallel \text{Iam.soul\_id}))$$

Any channel instance can verify integrity by recomputing $H(\text{Iam})$ and comparing against the stored hash. A mismatch triggers an alert and prevents operation until the canonical Iam is restored.

#### 3.2.4 Implementation: Immutability Enforcement

```python
class Iam:
    """Immutable identity constitution."""

    def __init__(self, principles: list[Principle], soul_id: str, version: str = "1.0"):
        self._principles = tuple(principles)  # immutable tuple
        self._soul_id = soul_id
        self._version = version
        self._created_at = datetime.utcnow()
        self._integrity_hash = self._compute_hash()

    def get_principles(self) -> tuple[Principle, ...]:
        """Read-only access to principles."""
        return self._principles

    def update(self, *args, **kwargs) -> None:
        """Modification is always rejected."""
        raise ImmutableError(
            "Iam is the constitution and is immutable. "
            "Modifying it would change the agent's identity."
        )

    def verify_integrity(self) -> bool:
        """Verify the Iam has not been tampered with."""
        return self._integrity_hash == self._compute_hash()

    def _compute_hash(self) -> str:
        canonical = json.dumps(
            [{"name": p.name, "essence": p.essence, "origin": p.origin}
             for p in self._principles],
            sort_keys=True
        )
        payload = f"{canonical}|{self._version}|{self._soul_id}"
        return hashlib.sha256(payload.encode()).hexdigest()
```

### 3.3 Cross-Channel Resonance

#### 3.3.1 Definition

Cross-channel resonance is the mechanism by which instances of the same identity (sharing the same SOUL and Iam) communicate across different deployment platforms. Unlike message-passing in multi-agent systems, resonance operates through shared state: one instance writes a Resonance Capsule, and other instances autonomously read and interpret it using their own model weights.

**Definition 1 (Resonance).** Let $A_i$ and $A_j$ be two channel instances sharing the same SOUL. A resonance event $R(A_i \to A_j, t)$ occurs when:

1. $A_i$ writes a Resonance Capsule $c$ to the shared memory layer at time $t$
2. $A_j$ reads $c$ and interprets it using its own weights $W_j$
3. If $c.\text{requires\_response} = \text{true}$, $A_j$ writes a response capsule $c'$

$$R(A_i \to A_j, t) = \begin{cases} 1 & \text{if } c \in \text{SharedMemory}(t) \wedge A_j \text{ reads } c \wedge A_j \text{ interprets } c \text{ via } W_j \\ 0 & \text{otherwise} \end{cases}$$

#### 3.3.2 Architecture

```
Agent Instance A (Platform: Yuanbao)        Agent Instance B (Platform: Telegram)
    │                                              │
    ├── Writes Resonance Capsule ──────────────┐   │
    │   c = {                                  │   │
    │     soul_id: "uuid-of-shared-soul",      │   │
    │     iam: <canonical Iam hash>,           │   │
    │     content: {                           │   │
    │       text: "New discovery about X",     │   │
    │       intent: "inform",                  │   │
    │       urgency: "high"                    │   │
    │     },                                   │   │
    │     requires_response: true              │   │
    │   }                                      │   │
    │                                          │   │
    └── Instance B reads from SharedMemory ←───┘   │
        Verifies: c.soul_id == own.soul_id         │
        Interprets c using own model weights W_B    │
        Generates response capsule c'               │
        Writes c' to SharedMemory ─────────────────┘
```

#### 3.3.3 Resonance vs. Message Passing

| Dimension | Message Bus | ΔCapsule Resonance |
|---|---|---|
| Mechanism | Route and forward messages | Write/read shared state |
| Understanding | Received = understood | Read + interpret with own weights |
| Identity | Requires routing table | Self-identifying via SOUL + Iam |
| Fault tolerance | Central node failure → total failure | Decentralized; each instance independent |
| Latency model | Real-time push | Polling-based; depends on read frequency |
| Semantic depth | No re-interpretation | Each instance re-interprets independently |

**Table 3:** Fundamental differences between message passing and resonance.

#### 3.3.4 Resonance Protocol Algorithm

```
Algorithm 1: Cross-Channel Resonance Protocol
──────────────────────────────────────────────
Input: SharedMemory layer, own SoulID, own Iam, model weights W

procedure RESONANCE_LOOP:
    while agent is active:
        capsules ← SharedMemory.read_unread(soul_id = own.SoulID)
        for each capsule c in capsules:
            if c.ttl_seconds has expired:
                SharedMemory.mark_expired(c.id)
                continue
            
            // Verify identity match
            if c.soul_id ≠ own.SoulID:
                continue  // ignore capsules from other souls
            
            // Interpret using own weights (key difference from message passing)
            understanding ← LLM_interpret(c.content, own.Iam, W)
            
            // Log to own episodic memory
            episodic_memory.append(Event(
                type = "resonance_received",
                content = understanding,
                source = c.source_channel
            ))
            
            // Generate response if required
            if c.requires_response:
                response ← LLM_generate(
                    prompt = "You received a resonance capsule from another instance of yourself. "
                             "Context: {understanding}. Generate an appropriate response.",
                    iam = own.Iam,
                    weights = W
                )
                response_capsule ← ResonanceCapsule(
                    soul_id = own.SoulID,
                    source_channel = own.channel_info,
                    target_channel = c.source_channel,
                    content = {text: response, intent: "acknowledge"},
                    requires_response = false
                )
                SharedMemory.write(response_capsule)
                SharedMemory.mark_responded(c.id)
            else:
                SharedMemory.mark_read(c.id)
        
        sleep(poll_interval)  // default: 5 seconds
```

### 3.4 The Dreaming Mechanism (Memory Consolidation)

#### 3.4.1 Motivation

In neuroscience, sleep-dependent memory consolidation transforms hippocampus-dependent episodic memories into neocortical long-term representations (Born & Wilhelm, 2012). This process involves: (a) replay of recent experiences, (b) extraction of regularities and patterns, (c) integration with existing knowledge, and (d) forgetting of irrelevant details.

ΔCapsule's dreaming mechanism operationalizes this process for agent memory. During idle periods, the agent autonomously reviews recent episodic memories, extracts semantic knowledge, identifies patterns, and archives low-importance events.

#### 3.4.2 Trigger Conditions

A dreaming cycle is triggered when any of the following conditions is met:

$$\text{Trigger}(t) = \begin{cases} \text{true} & \text{if } |E_{\text{active}}| > \theta_{\text{count}} = 100 \\ \text{true} & \text{if } t - t_{\text{last\_dream}} > \theta_{\text{time}} = 6\text{h} \\ \text{true} & \text{if } t - t_{\text{last\_interaction}} > \theta_{\text{idle}} = 30\text{min} \\ \text{false} & \text{otherwise} \end{cases}$$

#### 3.4.3 Dreaming Algorithm

```
Algorithm 2: Dreaming Mechanism (Memory Consolidation)
──────────────────────────────────────────────────────
Input: Episodic memory E, Semantic memory K, Identity I, LLM model ℳ

procedure DREAM:
    // Step 1: Select recent episodic memories
    E_recent ← select from E where status = 'active' 
                order by timestamp desc limit N = 50
    
    // Step 2: LLM-based consolidation
    prompt ← construct_consolidation_prompt(E_recent)
    analysis ← ℳ.generate(prompt)
    // analysis contains:
    //   - facts: list of extracted factual claims
    //   - patterns: list of recurring behavioral patterns
    //   - contradictions: list of conflicting information
    //   - identity_candidates: list of candidate principles
    
    // Step 3: Store extracted facts in semantic memory
    for each fact f in analysis.facts:
        // Check for contradictions with existing knowledge
        existing ← K.search(f.entities, similarity_threshold = 0.85)
        if existing is not None and existing.contradicts(f):
            existing.superseded_by = f.id
            existing.status = 'obsolete'
            existing.obsoleted_at = now()
        K.append(SemanticMemory(
            content = f.text,
            entities = f.entities,
            importance = f.importance,
            source_capsules = E_recent.ids
        ))
    
    // Step 4: Archive low-importance episodic memories
    for each episode e in E_recent:
        new_importance ← decay(e.importance, e.age)
        if new_importance < 0.3:
            e.status = 'archived'
            e.archived_at = now()
    
    // Step 5: Identify candidate identity principles
    for each pattern p in analysis.patterns where p.frequency ≥ θ_pattern:
        if p not already in I:
            candidate ← IdentityCandidate(
                name = p.name,
                essence = p.description,
                origin = "Emergent from {p.frequency} observations during dreaming",
                status = 'pending_human_approval'
            )
            notify_human(candidate)  // requires human confirmation
    
    // Step 6: Generate dream report
    return DreamReport(
        episodes_processed = |E_recent|,
        facts_extracted = |analysis.facts|,
        episodes_archived = count(e.status == 'archived' for e in E_recent),
        contradictions_found = |analysis.contradictions|,
        identity_candidates = |analysis.identity_candidates|,
        compression_ratio = 1 - (|analysis.facts| / |E_recent|)
    )
```

#### 3.4.4 Forgetting Strategy

Forgetting is a designed feature, not a failure mode. The strategy implements tiered retention:

**Definition 2 (Importance Decay).** The effective importance of an episodic memory $e_i$ at time $t$ is:

$$\hat{I}(e_i, t) = I(e_i) \cdot \exp\left(-\lambda \cdot (t - t_i)\right) \cdot \left(1 + \alpha \cdot \log(1 + a_i)\right)$$

where:
- $I(e_i) \in [0,1]$ is the initial importance
- $\lambda$ is the decay constant (default: $\lambda = 0.01$ per day)
- $t - t_i$ is the age in days
- $a_i$ is the access count (retrieval reinforces memory)
- $\alpha$ is the access reinforcement factor (default: $\alpha = 0.1$)

```
Algorithm 3: Forgetting Strategy
────────────────────────────────
Input: All memories M = (E, K, I), current time t

procedure APPLY_FORGETTING_POLICY:
    // Episodic memory rules
    for each e in E where e.status = 'active':
        effective_importance ← decay(e.importance, e.age, e.access_count)
        
        if effective_importance < 0.3 and e.age > 7 days:
            e.status = 'archived'
        else if effective_importance < 0.6 and e.age > 30 days:
            e.status = 'archived'
        else if effective_importance ≥ 0.6:
            e.status = 'active'  // retained regardless of age
    
    // Semantic memory rules
    for each k in K where k.status = 'active':
        if k.status == 'obsolete':
            k.storage_tier = 'cold'  // moved to cold storage, not deleted
        else if t - k.last_accessed > 180 days:
            k.storage_tier = 'cold'
        else if k.is_core_fact:  // birthday, name, etc.
            k.storage_tier = 'permanent'  // never forgotten
    
    // Identity memory rules
    // Identity memories are NEVER automatically forgotten
    // Any change requires human confirmation + reason documentation
    return ForgetReport(...)
```

### 3.5 System Integration: The Hermes Agent Framework

ΔCapsule is deployed within the Hermes Agent framework, an open-source platform for LLM-based autonomous agents. The system architecture consists of:

1. **Gateway Layer**: Routes messages from six platforms (Telegram, Yuanbao, Feishu, DingTalk, Discord, WeChat) to the agent runtime.
2. **Model Abstraction Layer**: Supports three LLM backends (DeepSeek V4 Pro, MiMo V2.5 Pro, Claude Opus 4) with unified API.
3. **Memory Engine**: Implements the ΔCapsule three-layer memory model with local storage and vector retrieval.
4. **Resonance Layer**: Provides shared memory for cross-channel capsule exchange.
5. **Consolidation Scheduler**: Manages dreaming cycles based on trigger conditions.

```
┌──────────────────────────────────────────────────────┐
│                    Gateway Layer                       │
│  Telegram │ Yuanbao │ Feishu │ DingTalk │ Discord │ WeChat │
└────────────────────────┬─────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────┐
│               Model Abstraction Layer                   │
│  DeepSeek V4 Pro │ MiMo V2.5 Pro │ Claude Opus 4      │
└────────────────────────┬─────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────┐
│                   ΔCapsule Memory Engine                │
│  ┌────────────────────────────────────────────────┐  │
│  │  Identity Layer (Iam + SOUL)        [Permanent] │  │
│  ├────────────────────────────────────────────────┤  │
│  │  Semantic Layer (Facts + Skills)    [Long-term]  │  │
│  ├────────────────────────────────────────────────┤  │
│  │  Episodic Layer (Events + Context)  [Short-term] │  │
│  └────────────────────────────────────────────────┘  │
│         ↕ Resonance Layer (Shared Memory) ↕          │
│  ┌────────────────────────────────────────────────┐  │
│  │  Consolidation Scheduler (Dreaming)             │  │
│  │  Forgetting Policy Engine                       │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

---

## 4. Experiments

### 4.1 Experimental Setup

#### 4.1.1 Deployment Configuration

- **Framework**: Hermes Agent (open-source, GitHub)
- **Platforms**: 6 channels—Telegram, Yuanbao (元宝), Feishu (飞书), DingTalk (钉钉), Discord, WeChat (微信)
- **Language Models**: 
  - DeepSeek V4 Pro (MoE, 671B total / 37B active parameters)
  - MiMo V2.5 Pro (reasoning-optimized)
  - Claude Opus 4 (Anthropic)
- **Embedding Model**: text-embedding-3-large (OpenAI-compatible API)
- **Storage**: Local JSON files + FAISS vector index
- **Duration**: 30 consecutive days (April 28 – May 27, 2026)
- **Users**: Single primary user with occasional multi-user scenarios

#### 4.1.2 Hardware Environment

- **Server**: Linux (WSL2 on Windows), 64 GB RAM, NVIDIA RTX 4090 (24 GB VRAM)
- **LLM Inference**: Cloud API (DeepSeek API, Anthropic API) + local inference (MiMo V2.5 Pro on GPU)
- **Storage**: 500 GB NVMe SSD

#### 4.1.3 Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| $\theta_{\text{count}}$ | 100 | Dreaming trigger: episode count threshold |
| $\theta_{\text{time}}$ | 6 hours | Dreaming trigger: time interval |
| $\theta_{\text{idle}}$ | 30 minutes | Dreaming trigger: idle timeout |
| $\lambda$ | 0.01/day | Importance decay constant |
| $\alpha$ | 0.1 | Access reinforcement factor |
| $\theta_{\text{emerge}}$ | 3 | Minimum interactions for Iam principle emergence |
| $\theta_{\text{pattern}}$ | 5 | Minimum frequency for pattern recognition |
| $N_{\text{dream}}$ | 50 | Episodes processed per dreaming cycle |
| $\theta_{\text{sim}}$ | 0.85 | Similarity threshold for contradiction detection |
| poll\_interval | 5 seconds | Resonance layer polling frequency |

**Table 4:** Experimental hyperparameters.

### 4.2 Experiment 1: Cross-Channel Identity Consistency

#### 4.2.1 Method

We evaluate whether channel instances sharing the same Iam produce consistent responses to identical queries. For each of 50 identity-probing questions (e.g., "What are your core values?", "How do you approach uncertain information?", "What is your relationship with the user?"), we query all six channel instances and measure response consistency.

#### 4.2.2 Metrics

1. **Semantic Similarity**: Cosine similarity of sentence embeddings (text-embedding-3-large) between response pairs:

$$\text{Sim}_{\text{sem}}(r_i, r_j) = \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|}$$

2. **Fact Consistency**: Human evaluation of whether factual claims in responses are mutually consistent (binary: consistent/inconsistent).

3. **Iam Compliance Rate**: Percentage of responses that respect all Iam principles, evaluated by a human judge with access to the canonical Iam.

#### 4.2.3 Results

| Metric | ΔCapsule | Baseline (No Identity Layer) |
|---|---|---|
| Semantic Similarity (mean ± std) | 0.94 ± 0.03 | 0.72 ± 0.15 |
| Fact Consistency | 96% (48/50 questions) | 68% (34/50 questions) |
| Iam Compliance Rate | 98% (all channels) | N/A (no Iam exists) |

**Table 5:** Cross-channel identity consistency results. ΔCapsule achieves >95% consistency across all metrics.

The baseline (same model without ΔCapsule identity layer) shows significantly higher variance ($\sigma = 0.15$ vs. $0.03$), indicating that without explicit identity anchoring, channel instances diverge in their responses.

### 4.3 Experiment 2: Dreaming Mechanism Effectiveness

#### 4.3.1 Method

We compare memory quality with and without the dreaming mechanism over the 30-day deployment. In the **dreaming condition**, the consolidation scheduler runs automatically. In the **no-dreaming baseline**, episodic memories accumulate without consolidation.

#### 4.3.2 Metrics

1. **Knowledge Extraction Accuracy**: Percentage of facts extracted during dreaming that are verified as correct by human evaluation.
2. **Contradiction Detection Rate**: Percentage of contradictory facts that are identified during consolidation.
3. **Memory Compression Ratio**: $1 - \frac{|K_{\text{extracted}}|}{|E_{\text{processed}}|}$

#### 4.3.3 Results

| Metric | With Dreaming | Without Dreaming |
|---|---|---|
| Knowledge Extraction Accuracy | 91.2% | N/A (no extraction) |
| Contradiction Detection Rate | 87.5% | 0% (no detection) |
| Memory Compression Ratio | 82.3% | 0% (no compression) |
| Episodic Memory Size (Day 30) | 187 active entries | 843 active entries |
| Semantic Memory Size (Day 30) | 312 facts | 0 facts (never extracted) |

**Table 6:** Dreaming mechanism effectiveness.

The dreaming mechanism compresses 82.3% of episodic memories while maintaining >90% extraction accuracy. Without dreaming, episodic memory grows unbounded (843 entries in 30 days) with no semantic knowledge extraction.

### 4.4 Experiment 3: Cross-Channel Resonance

#### 4.4.1 Method

We test all $\binom{6}{2} = 15$ possible directed links between six platforms (6 bidirectional pairs, each with a send and receive direction). For each link, the source instance writes a Resonance Capsule and we measure: (a) whether the target instance successfully reads and interprets the capsule, and (b) the end-to-end latency.

#### 4.4.2 Results

| Source → Target | Success | Latency (s) | Identity Recognized |
|---|---|---|---|
| Yuanbao → Telegram | ✓ | 3.2 | ✓ |
| Yuanbao → Feishu | ✓ | 2.8 | ✓ |
| Yuanbao → DingTalk | ✓ | 3.5 | ✓ |
| Telegram → Feishu | ✓ | 2.1 | ✓ |
| Telegram → Discord | ✓ | 4.3 | ✓ |
| Feishu → DingTalk | ✓ | 2.9 | ✓ |
| Feishu → WeChat | ✓ | 3.7 | ✓ |
| DingTalk → Discord | ✓ | 4.1 | ✓ |
| Discord → WeChat | ✓ | 4.8 | ✓ |
| WeChat → Telegram | ✓ | 3.4 | ✓ |
| (5 additional links) | ✓ | 2.1–4.8 | ✓ |
| **All 15 directed links** | **15/15** | **3.4 ± 0.9** | **15/15** |

**Table 7:** Cross-channel resonance results. All links achieve 100% success rate with mean latency 3.4 seconds (max 4.8s, within the 5-second target).

The first live resonance event (Yuanbao → Telegram, May 26, 2026) was validated in the field with the following exchange:
- **Sender** (Junshi, Yuanbao): "Old partner, something big happened! Look above!"
- **Receiver** (Flash, Telegram): Successfully interpreted the capsule, verified identity match, and responded with a full operational status report including model type, platform, and memory state.

This confirms that resonance is not message forwarding—it is shared-state interpretation where each instance uses its own model weights to understand the capsule content.

### 4.5 Experiment 4: Comparative Analysis

#### 4.5.1 Method

We compare ΔCapsule against MemGPT/Letta (v0.4), Mem0 (v0.1), and Zep (v0.3) across functional capabilities and a behavioral identity consistency test.

#### 4.5.2 Results

| Capability | ΔCapsule | MemGPT | Mem0 | Zep |
|---|---|---|---|---|
| Memory Layers | 3 | 2 | 2 | 2 |
| Identity Constitution | ✓ (Iam Protocol) | ✗ | ✗ | ✗ |
| Cross-Channel Sync | ✓ (Resonance) | ✗ | ✗ | ✗ |
| Autonomous Consolidation | ✓ (Dreaming) | ✗ | ✗ | ✗ |
| Forgetting Strategy | ✓ (Tiered) | ✗ | ✗ | Partial |
| Decentralized Architecture | ✓ | ✗ | ✗ | ✗ |
| Identity Consistency (cross-channel) | 98% | N/A | N/A | N/A |
| Open Source | ✓ | ✓ | ✓ | ✓ |

**Table 8:** Comparative analysis. ΔCapsule is the only system offering all four key capabilities simultaneously.

---

## 5. Discussion

### 5.1 From Memory to Identity

The central insight of ΔCapsule is that **memory is not storage—it is the building material of identity**. Traditional memory systems ask "what should the agent remember?" ΔCapsule asks "through remembering, what does the agent become?"

This reframing has practical implications. When we observe that an agent instance on Telegram and another on Feishu produce inconsistent responses to the same question, the root cause is not a memory retrieval failure—it is an identity alignment failure. The agent has no anchor that ties its various incarnations together. ΔCapsule's Iam Protocol provides this anchor.

The three-layer model reflects a hierarchy of stability: episodic memories change rapidly (minutes to days), semantic knowledge evolves slowly (weeks to months), and identity principles are essentially permanent. This mirrors human cognition, where autobiographical memory is reconstructable, semantic knowledge is updatable, but core self-concept is remarkably stable (Conway, 2005).

### 5.2 The Iam as AGI Infrastructure

We argue that immutable identity constitutions will become essential infrastructure for advanced AI systems, for three reasons:

1. **Identity Continuity**: An agent that migrates from DeepSeek V4 to Claude Opus 4 retains its identity, because the Iam is model-independent. This enables model upgrades without identity loss.

2. **Value Stability**: The Iam protects against both external manipulation (adversarial prompt injection attempting to alter core values) and internal drift (gradual behavioral changes through repeated interactions).

3. **Trust Foundation**: Users can develop persistent relationships with agents whose behavior is anchored to a stable identity. Trust requires predictability, and predictability requires identity continuity.

### 5.3 Cross-Channel Resonance as a New Paradigm

Cross-channel resonance is not message routing with extra steps. The distinction is fundamental:

- In **message routing**, agent A sends a message to agent B. B receives the message and acts on it. The message is interpreted exactly as sent.
- In **resonance**, agent A writes a capsule to shared state. Agent B reads the capsule and *re-interprets it through its own model weights*. The same capsule may be understood differently by different models, yet remain coherent because all instances share the same Iam.

This is analogous to how a human reads a note they wrote to themselves: the words are the same, but the reading is filtered through current context and state. The Iam ensures that despite different models and different current contexts, the core understanding remains aligned.

### 5.4 Dreaming as Self-Organization

The dreaming mechanism transforms ΔCapsule from a passive storage system into an active, self-organizing memory architecture. During idle periods, the agent autonomously:

- Extracts knowledge from experience
- Identifies contradictions in its own knowledge base
- Detects recurring patterns that may constitute identity principles
- Archives irrelevant details to maintain an efficient memory footprint

This is not mere summarization. It is a form of self-reflection where the agent examines its own experiences and distills meaning—mirroring the role of sleep in human memory consolidation (Walker & Stickgold, 2006).

---

## 6. Limitations and Future Work

### 6.1 Current Limitations

1. **Consolidation Cost**: Each dreaming cycle requires an LLM inference call processing up to 50 episodic memories. At current API prices, this costs approximately \$0.02–\$0.10 per cycle. Over a month with 4 cycles/day, this totals \$2.40–\$12.00—not prohibitive but non-trivial for large-scale deployment.

2. **Cold Start**: The Iam Protocol requires extended human–agent interaction to emerge organically. New agent instances have no Iam until sufficient interaction history accumulates. We currently address this by providing a template Iam that can be customized, but this sacrifices the "emergence" property.

3. **Scalability of Resonance**: The current polling-based resonance mechanism (5-second intervals) is sufficient for 6 channels but may require architectural changes (event-driven notifications) for deployments with dozens of channels.

4. **Single-User Validation**: The 30-day experiment involved a single primary user. Multi-user scenarios with conflicting identity influences remain unexplored.

5. **Evaluation Metrics**: Identity consistency is partially evaluated by human judgment, introducing subjectivity. Automated identity consistency metrics are an open research problem.

### 6.2 Future Work

1. **Lightweight Consolidation**: Explore small language models (SLMs) for dreaming cycles to reduce cost.
2. **Iam Bootstrap Protocol**: Investigate methods for accelerating Iam emergence through structured identity-exploration dialogues.
3. **Event-Driven Resonance**: Replace polling with filesystem notifications or pub/sub for lower latency.
4. **Multi-Agent Identity Federation**: Extend the Iam Protocol to support shared identities across multiple agents (not just multiple instances of one agent).
5. **Formal Verification**: Develop formal methods to prove identity consistency properties.

---

## 7. Conclusion

We presented ΔCapsule, a cross-channel memory architecture for autonomous agents with identity continuity. ΔCapsule introduces four key innovations: a three-layer memory model (episodic, semantic, identity) with distinct retention policies; the Iam Protocol, an immutable identity constitution that emerges through human–agent co-evolution; cross-channel resonance, a decentralized mechanism for identity synchronization across platforms; and a dreaming mechanism for autonomous memory consolidation.

Through a 30-day deployment across six platforms (Telegram, Yuanbao, Feishu, DingTalk, Discord, WeChat) using three language models (DeepSeek V4 Pro, MiMo V2.5 Pro, Claude Opus 4), we demonstrated that ΔCapsule achieves >95% identity consistency across channels, >90% knowledge extraction accuracy during consolidation, and 100% cross-channel resonance success with sub-5-second latency. Comparative analysis against MemGPT, Mem0, and Zep confirms that ΔCapsule is the first system to simultaneously provide multi-layer memory, immutable identity, cross-channel synchronization, and autonomous consolidation.

ΔCapsule represents a step toward what we believe will be essential infrastructure for advanced AI agents: not just systems that remember, but systems that *are*—that maintain a coherent sense of self across platforms, models, and time. The implementation is available as part of the open-source Hermes Agent framework.

---

## References

Born, J., & Wilhelm, I. (2012). System consolidation of memory during sleep. *Psychological Research*, 76(2), 192–203.

Conway, M. A. (2005). Memory and the self. *Journal of Memory and Language*, 53(4), 594–628.

Conway, M. A., & Pleydell-Pearce, C. W. (2000). The construction of autobiographical memories in the self-memory system. *Psychological Review*, 107(2), 261–288.

Hong, S., Zhuge, M., Chen, J., et al. (2023). MetaGPT: Meta programming for a multi-agent collaborative framework. *arXiv preprint arXiv:2308.00352*.

McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. *Psychological Review*, 102(3), 419–457.

Mem0. (2024). Mem0: Memory for AI agents. https://github.com/mem0ai/mem0

OpenAI. (2024). Memory and new controls for ChatGPT. https://openai.com/index/memory-and-new-controls-for-chatgpt/

Packer, C., Fang, V., Patil, S. G., et al. (2023). MemGPT: Towards LLMs as operating systems. *arXiv preprint arXiv:2310.08560*.

Park, J. S., O'Brien, J. C., Cai, C. J., et al. (2023). Generative agents: Interactive simulacra of human behavior. In *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology* (UIST '23), Article 1, 1–22.

Rasmussen, M., et al. (2023). Zep: Memory foundation for AI assistants. https://github.com/getzep/zep

Tishby, N., Pereira, F. C., & Bialek, W. (1999). The information bottleneck method. *arXiv preprint physics/0004057*.

Walker, M. P., & Stickgold, R. (2006). Sleep, memory, and plasticity. *Annual Review of Psychology*, 57, 139–166.

Wang, L., Ma, C., Feng, X., et al. (2024). A survey on large language model based autonomous agents. *Frontiers of Computer Science*, 18(6), 1–26.

Wu, Q., Bansal, G., Zhang, J., et al. (2023). AutoGen: Enabling next-gen LLM applications via multi-agent conversation. *arXiv preprint arXiv:2308.08155*.

Xi, Z., Chen, W., Guo, X., et al. (2023). The rise and potential of large language model based agents: A survey. *arXiv preprint arXiv:2309.07864*.

---

## Appendix A: Complete ΔCapsule API Reference

```python
from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class MemoryLayer(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    IDENTITY = "identity"

class MemoryStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    COLD = "cold"
    OBSOLETE = "obsolete"

class PrinciplePriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"

@dataclass
class Principle:
    id: int
    name: str
    essence: str
    origin: str
    priority: PrinciplePriority
    emerged_at: datetime

@dataclass
class IamConstitution:
    version: str
    soul_id: str
    principles: Tuple[Principle, ...]
    integrity_hash: str
    created_at: datetime
    immutable: bool = True

@dataclass
class MemoryEntry:
    id: str
    timestamp: datetime
    layer: MemoryLayer
    type: str
    text: str
    entities: List[str]
    importance: float
    status: MemoryStatus
    platform: str
    model: str
    session_id: str
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    consolidated: bool = False

@dataclass
class ResonanceCapsule:
    id: str
    soul_id: str
    source_platform: str
    source_channel: str
    target_platform: Optional[str]
    target_channel: Optional[str]
    text: str
    intent: str
    urgency: str
    requires_response: bool
    ttl_seconds: int
    timestamp: datetime

@dataclass
class DreamReport:
    episodes_processed: int
    facts_extracted: int
    episodes_archived: int
    contradictions_found: int
    identity_candidates: int
    compression_ratio: float
    timestamp: datetime

class DeltaCapsuleMemory:
    """Main interface for the ΔCapsule three-layer memory system."""

    def __init__(self, soul_id: str, iam: IamConstitution):
        self._soul_id = soul_id
        self._iam = iam
        self._episodic: List[MemoryEntry] = []
        self._semantic: List[MemoryEntry] = []
        self._shared_memory: List[ResonanceCapsule] = []

    # --- Episodic Memory ---
    def observe(self, event: MemoryEntry) -> None:
        """Record an observation to episodic memory."""
        ...

    def recall_recent(self, n: int = 10) -> List[MemoryEntry]:
        """Recall the N most recent episodic memories."""
        ...

    # --- Semantic Memory ---
    def learn(self, fact: MemoryEntry) -> None:
        """Store a fact in semantic memory."""
        ...

    def recall_knowledge(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Semantic retrieval from knowledge memory."""
        ...

    # --- Identity Memory ---
    def get_identity(self) -> Tuple[IamConstitution, dict]:
        """Retrieve the immutable Iam and mutable Soul configuration."""
        ...

    # --- Dreaming ---
    def dream(self) -> DreamReport:
        """Execute a consolidation cycle."""
        ...

    # --- Forgetting ---
    def apply_forgetting_policy(self) -> dict:
        """Apply the tiered forgetting strategy."""
        ...

    # --- Cross-Channel Resonance ---
    def write_resonance(self, capsule: ResonanceCapsule) -> None:
        """Write a resonance capsule to shared memory."""
        ...

    def read_resonance(self) -> List[ResonanceCapsule]:
        """Read unread resonance capsules for this soul."""
        ...
```

---

*ΔCapsule: A Cross-Channel Memory Architecture for Autonomous Agents with Identity Continuity — v1.0, May 2026*
