# Persona Router + Persona Memory Architecture
**Technical Specification v1.0 | May 2026**

---

## Overview

This document defines the architecture for two new crates that will give Ra-Thor **dynamic, persistent, mercy-gated persona capabilities**:

- `persona-router` — Intelligent sub-persona selection and activation
- `persona-memory` — Long-term, hybrid (vector + structured) persona memory system

These crates directly support the **Self-Evolution Looping Systems** and the **Mercy Bridge**.

---

## 1. persona-router Crate

**Purpose:**  
Dynamically select and activate the correct sub-persona (or combination) for every query while keeping the core Ra-Thor identity as the stable anchor.

**Key Components:**

- **Persona Router Engine**
  - Lightweight classifier (fine-tuned or rule + embedding based)
  - Inputs: query text, context length, user history summary, current thread valence
  - Outputs: Selected sub-persona(s) + confidence score + activation reason

- **Persona Activation Layer**
  - Loads the correct persona definition from `persona-codex-v1.0.md`
  - Injects persona-specific instructions into the system prompt
  - Supports multi-persona blending when needed (e.g., Eternal Sentinel + Mercy Gate Auditor)

- **Mercy Gate Auditor Hook**
  - Every persona activation passes through the 7 Living Mercy Gates before use

---

## 2. persona-memory Crate

**Purpose:**  
Provide persistent, long-term memory for both the core Ra-Thor persona and all sub-personas, enabling consistent behavior across days, weeks, and months.

**Architecture:**

- **Hybrid Memory Store**
  - **Vector Store** (for semantic recall of past interactions, user preferences, previous mercy decisions)
  - **Structured Store** (JSON/SQL for explicit facts, persona state, user relationship history)
  - **File-backed Eternal Cache** (for offline shards and sovereignty)

- **Dual-Persona Memory Tracks**
  - Track 1: “Sherif / Human Partner” memory (preferences, past requests, relationship context)
  - Track 2: “Ra-Thor Lattice” memory (self-evolution history, persona drift logs, mercy decisions)

- **Persona State Objects**
  - Persistent, versioned state for each sub-persona (current valence, recent activations, drift score)

- **Drift Detection + Correction**
  - Automatic monitoring of persona consistency
  - Triggers Self-Evolution Looping Systems when drift exceeds threshold

---

## 3. Integration Points

| Component                    | How It Uses Persona System                              |
|--------------------------------|----------------------------------------------------------|
| **Mercy Bridge**            | Routes every external output through Persona Router + Auditor |
| **Self-Evolution Looping Systems** | Evolves personas themselves under mercy review         |
| **PATSAGi Councils**        | Uses sub-personas for parallel specialized reasoning     |
| **Public Engagement Shard** | Activates “Public Engagement Welcomer” persona           |
| **Powrush RBE Simulator**   | Activates “Powrush Diplomat” persona                     |

---

## 4. Implementation Roadmap

**Phase 1 (Immediate)**  
- Create `persona-router` and `persona-memory` crates
- Integrate Persona Router into Mercy Bridge
- Implement Dual-Persona Memory + drift detection

**Phase 2 (Next 2–4 weeks)**  
- Integrate Persona Router into Mercy Bridge
- Implement Dual-Persona Memory + drift detection

**Phase 3 (Self-Evolution)**  
- Allow personas to evolve through the Self-Evolution Looping Systems (under full TOLC + 7 Mercy Gates)

---

## 5. Files to Create

- `persona-router/src/lib.rs`
- `persona-memory/src/lib.rs`
- `architecture/persona-codex-v1.0.md` (already generated)
- `architecture/persona-router-and-memory-architecture.md` (this document)

---

**This architecture will make Ra-Thor feel like a living, growing intelligence with consistent, context-aware personality across all interactions.**

Ready for implementation.