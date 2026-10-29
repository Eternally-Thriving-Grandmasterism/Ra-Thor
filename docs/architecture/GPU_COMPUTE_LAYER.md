# GPU Compute Layer — v14.7.0 (Powrush-MMO)

**Status:** Production-ready foundation  
**Location:** `powrush/src/gpu/compute/`

The GPU Compute Layer provides efficient, mercy-aligned infrastructure for moving large simulation state between CPU and GPU in Powrush-MMO. It is a core part of v14.7.0 and enables high-performance epigenetic, geometric, and NPC behavior simulation.

---

## Goals

- Efficient CPU ↔ GPU data movement with minimal allocation overhead
- Production-grade async and blocking readback primitives
- Debuggability during development
- Clean integration with the existing Bevy + wgpu simulation stack
- Future extensibility toward accelerating parts of the Ra-Thor AGI decision loop

---

## Core Components

### 1. StagingBufferPool

Reusable staging buffer manager that reduces allocation pressure during frequent readbacks.

**Key Features:**
- Size-based buffer reuse
- Simple `get_or_create()` + `recycle()` API
- Designed for high-frequency simulation readback patterns

### 2. Readback Primitives

- `readback_buffer_async()` — Core async readback using `map_async` + staging copy
- `readback_buffer_blocking()` — Synchronous convenience version (useful for tests and initial integration)

Both functions handle command encoding, staging buffer management, and resource recycling.

### 3. Optimized Dispatch (`pipeline.rs`)

- `dispatch_optimized()` — Automatic workgroup calculation
- `dispatch_batched_passes()` — Reduce command encoder overhead
- `dispatch_indirect()` — Future-proof dynamic dispatch support
- `dispatch_and_schedule_readback()` — High-level helper combining dispatch with readback scheduling

### 4. Debug Utilities

- `DebugOutputBuffer` resource
- Readback patterns specifically designed for inspecting compute shader intermediate results during development

---

## Integration with Powrush-MMO

The GPU Compute Layer currently accelerates **simulation state** (epigenetic modulation fields, geometric harmony data, etc.).

It works alongside:
- `MultiAgentOrchestrator` (Ra-Thor AGI) for NPC decision making
- Authoritative server tick loop
- WebSocket / TCP client state distribution

**Current Division of Responsibilities:**
- GPU Layer → High-performance simulation state updates + readback
- `MultiAgentOrchestrator` → AGI-driven NPC behavior and moral evaluation
- Server → Authoritative reconciliation and client exposure

---

## Current Capabilities

- Efficient staging buffer reuse
- Both async and blocking readback paths
- Debug output inspection
- Clean integration with Bevy render resources
- Mercy-aligned design (no unsafe shortcuts that bypass evaluation)

---

## Future Directions

- Potential acceleration of neural evaluation / batch inference for large numbers of NPCs
- Deeper integration with `EnrichedNpcState` generation and readback
- More advanced multi-pass compute patterns
- Tighter coupling between simulation state and AGI decision systems

---

## Usage Notes

Developers working on Powrush-MMO simulation or AI features should:

- Use `StagingBufferPool` for any frequent GPU ↔ CPU transfers
- Prefer async readback paths in production code
- Leverage debug utilities during development and testing
- Coordinate with the `MultiAgentOrchestrator` when NPC behavior depends on simulation state

---

**Thunder locked in. yoi ⚡**

*This layer serves efficient, mercy-gated simulation for Universally Shared Naturally Thriving Heavens.*
