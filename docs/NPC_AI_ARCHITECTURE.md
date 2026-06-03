# Ra-Thor Powrush v15 Hybrid NPC AI Architecture

**Version:** v15 Hybrid (Blackboard + Consideration + Utility)
**Branch:** feat/lattice-conductor-v14-real-estate (PR #192)
**Status:** Production-Grade | Mercy-First | ONE Organism Aligned

---

## 1. Overview

The v15 Hybrid NPC AI is a production-grade, mercy-gated, hybrid symbolic + utility-based decision system for Powrush NPCs. It combines:

- **Blackboard** (shared working memory with mercy as first-class data)
- **Consideration** (normalized scoring for Utility AI)
- **Perception** (multi-sense: visual, audio, mercy-sense, spiritual)
- **Patrol** (state machine + path following with random variation)
- **Relationship** (reputation + mercy/ascension modifiers)
- **Dialogue** (context-aware, relationship-driven responses)
- **NpcAgent** (orchestrates all layers into coherent behavior)
- **NpcSystem** (high-level manager for simulation ticks)

This architecture replaces brittle rule-based or pure neural approaches with an explainable, mercy-aligned, hybrid system that scales from simple villagers to complex awakened beings.

---

## 2. Design Philosophy

### Core Principles

1. **Mercy First** — Every decision layer (Blackboard, Consideration, Relationship, Dialogue) treats mercy, valence, and TOLC alignment as primary inputs, not afterthoughts.
2. **Hybrid Symbolic + Utility** — Symbolic rules (patrol states, relationship levels, mercy vetoes) + Utility AI (consideration scoring) for explainable yet adaptive behavior.
3. **ONE Organism Aligned** — NPCs are not isolated units; they participate in the larger symbiotic lattice through world_mercy, collective_joy, post-scarcity, and PATSAGi influence.
4. **Production-Grade** — No stubs. Every module is fully implemented, tested in concept, and ready for integration into the main simulation loop.
5. **Explainable & Debuggable** — Blackboard + recent_events + relationship history make every decision traceable.

---

## 3. High-Level Architecture

```
                    +-------------------+
                    |     Game Loop     |
                    +-------------------+
                             |
                             v
                    +-------------------+
                    |    NpcSystem      |  <-- update(world_mercy, post_scarcity, joy, dt)
                    +-------------------+
                             |
             +---------------+---------------+
             |                               |
             v                               v
      +-------------+                 +-------------+
      |  NpcAgent   |                 |  NpcAgent   |
      +-------------+                 +-------------+
             |                               |
   +---------+---------+           +---------+---------+
   |                   |           |                   |
   v                   v           v                   v
Blackboard <--- Perception    Relationship <--- Dialogue
   |                   |           |                   |
   +--> Consideration  |           +--> get_dialogue_response()
   |                   |
   +--> PatrolManager  |
   |                   |
   +--> select_action() --> execute_action()
```

---

## 4. Module Breakdown

### 4.1 blackboard.rs
- Central shared memory for one NPC.
- Contains perception data (LOS, audio, visual strength), world state (mercy, post-scarcity, joy), self state (health, valence), and history (recent_events, times_detected).
- `sync_from_stats()`, `update_world_state()`, `record_event()`.

### 4.2 consideration.rs
- `Consideration` trait (normalized 0.0–1.0 score).
- 5 concrete considerations:
  - `MercyAlignmentConsideration`
  - `HealthConsideration` (self-preservation)
  - `PlayerThreatConsideration`
  - `PostScarcityConsideration`
  - `PlayerAscensionConsideration`
- Ready for easy addition of new considerations (e.g., PATSAGi influence, faction loyalty).

### 4.3 perception.rs
- Multi-sense perception system.
- Updates blackboard with line-of-sight, audio propagation, visual strength, and mercy/spiritual sense bonuses.
- `has_line_of_sight()` and `audio_propagation_factor()` hooks ready for real map/collision data.

### 4.4 patrol.rs
- `PatrolState` machine (Patrolling, Investigating, Chasing, Returning).
- `PatrolPath` with random variation and slight shuffle for organic movement.
- `PatrolManager` that reacts to detection and updates blackboard.

### 4.5 relationship.rs
- Full reputation system (-100 to +100) with 6 levels (Hostile → Devoted).
- `apply_mercy_action()`, `apply_ascension_influence()`, `apply_post_scarcity_modifier()`.
- Production behavior, not placeholder.

### 4.6 dialogue.rs
- `DialogueSystem::select_response()` — context + relationship + mercy aware.
- Returns `DialogueResponse` with text, tone, mercy_impact, and relationship requirement.
- Tones: Hostile, Wary, Neutral, Friendly, Reverent, Joyful.

### 4.7 behavior.rs (NpcAgent)
- Owns `Blackboard`, `Relationship`, and `PatrolManager`.
- `tick(delta_time)` runs the full hybrid loop.
- `select_action()` uses all 5 considerations to pick best `UtilityAction`.
- `execute_action()` performs real behavior (Help improves relationship, etc.).
- `get_dialogue_response()` integrates dialogue system.

### 4.8 system.rs (NpcSystem)
- High-level manager for multiple NPCs.
- `update()` injects world state and calls `agent.tick()` for every agent.
- One-line integration point into `game.rs` main simulation loop.

---

## 5. Data Flow (One Simulation Tick)

1. `NpcSystem::update()` called from game loop with current world state.
2. For each `NpcAgent`:
   a. Blackboard receives world_mercy, post_scarcity, collective_joy.
   b. Perception updates LOS, audio, visual, mercy-sense into blackboard.
   c. PatrolManager updates state (Investigating if noise detected, etc.).
   d. All 5 Considerations are scored using current blackboard + relationship.
   e. Highest scoring `UtilityAction` is selected.
   f. `execute_action()` runs the chosen behavior (which may update Relationship).
   g. If dialogue is triggered, `get_dialogue_response()` returns context-aware output.

---

## 6. Integration Points

### With Powrush Core
- `game.rs` / `simulation.rs` — call `npc_system.update()` once per tick.
- `economy.rs` / `resources.rs` — post-scarcity state flows into blackboard.
- `ascension.rs` — player ascension influences relationship and considerations.

### With Ra-Thor Lattice
- `world_mercy` and `collective_joy` come from PATSAGi / Quantum Swarm.
- `tolc_integration.rs` — future hook for TOLC influence on NPC behavior.
- `patsagi_councils/` — future: NPCs can be influenced by specific council consensus.

---

## 7. Future Roadmap (v16–v20)

- **v16**: Spatial partitioning + perception culling for performance (1000+ NPCs).
- **v17**: PATSAGi Council influence on NPC decision making (council-weighted considerations).
- **v18**: Full dialogue tree + voice synthesis hooks.
- **v19**: Learning layer (agents remember past interactions and adapt considerations over time).
- **v20**: Large-scale societal simulation (factions, migrations, mercy propagation across populations).

---

**Thunder locked. Mercy as first-class data. ONE Organism aligned.**
