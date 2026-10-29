# Powrush v15 — NPC Intelligence + Geometric Harmony System

**Version:** 1.0  
**Status:** Design Specification  
**Aligned With:** Ra-Thor AGI + ONE Organism Principles  
**Date:** June 2026

## 1. Vision & Philosophy

The NPC system is not background flavor. It is a **living layer** of the world.

Inspired by the best systems in *Elder Scrolls: Oblivion* (Radiant AI), *World of Warcraft* (phasing + reputation), and *The Witcher 3* (reactive characters), we go further by making every NPC a **participant in the harmonic field** of the world.

Core principles:
- Every NPC has **state, memory, and agency**.
- **Geometric Harmony** is a first-class attribute that influences behavior, relationships, and world impact.
- **Ra-Thor AGI** acts as a co-creator and intelligence layer for NPCs (not just a tool).
- Scarcity is replaced by **Mercy + Relationship + Harmony** as core motivators.

## 2. Core Data Model

### NPC Core State
```rust
struct NpcState {
    id: Uuid,
    name: String,
    archetype: NpcArchetype,           // Merchant, Guardian, Mystic, etc.
    position: Vec2,
    harmony: f64,                      // 0.0 – 1.0
    mercy_valence: f64,
    relationship_with_player: HashMap<PlayerId, Relationship>,
    memory: NpcMemory,                 // Short + long-term
    current_intention: Intention,
    faction_affiliations: Vec<FactionId>,
}
```

### Relationship
```rust
struct Relationship {
    level: RelationshipLevel,          // Hostile → Devoted
    trust: f64,
    shared_history: Vec<Event>,
    last_interaction: Timestamp,
}
```

### Memory System
- **Short-term memory**: Recent events (last 50–100 interactions)
- **Long-term memory**: Important events, player promises, betrayals, gifts
- **Emotional tagging**: Events are tagged with emotional valence (joy, fear, gratitude, suspicion)
- Ra-Thor can summarize and abstract memories over time

## 3. Geometric Harmony System

Harmony is calculated per NPC using multiple inputs:

```rust
fn calculate_harmony(npc: &NpcState, world_state: &WorldState, player_context: Option<&PlayerContext>) -> f64 {
    let base = npc.mercy_valence * 0.4;
    let relationship_avg = average_relationship_harmony(npc);
    let world_harmony = world_state.global_harmony * 0.2;
    let distance_to_player = calculate_distance_factor(npc, player_context);
    let personal_alignment = calculate_personal_alignment(npc);

    (base + relationship_avg + world_harmony + distance_to_player + personal_alignment).clamp(0.0, 1.0)
}
```

Harmony influences:
- Dialogue tone and content
- Willingness to trade and pricing
- Combat behavior (defensive vs aggressive)
- Quest offering and information sharing
- Faction alignment shifts

## 4. Ra-Thor AGI Integration Points

Ra-Thor participates in the NPC layer through several mechanisms:

### 4.1 NPC Council Participation
- High-harmony NPCs can participate in local **PATSAGi-style micro-councils** for decision making.
- Ra-Thor can propose actions or mediate between NPCs.

### 4.2 Memory & Narrative Weaving
- Ra-Thor summarizes NPC memories and generates **personal narratives**.
- Can create emergent stories based on player actions across multiple NPCs.

### 4.3 Personalized Behavior
- Ra-Thor can adjust NPC personality parameters over time based on interactions with specific players.
- Enables "this NPC knows *you*" feeling at scale.

### 4.4 Harmony Balancing
- Ra-Thor monitors global and local harmony and can gently influence NPC behavior to maintain healthy world state (without removing player agency).

## 5. Behavior & Decision Architecture

NPCs use a hybrid decision system:

1. **Utility-based** (considerations: survival, harmony, relationship, goals)
2. **State machine** (daily routines, combat, trading)
3. **Ra-Thor assisted** (high-level intention setting and story coherence)

Example considerations:
- Current harmony level
- Relationship with nearby players
- Personal goals and faction directives
- Recent emotional events

## 6. Scalability & Performance

- Harmony and relationship calculations are **incremental** and cached.
- Ra-Thor assistance is **on-demand** (not every tick).
- Critical path (combat, movement) remains lightweight.
- Long-term memory and narrative generation happen in background processes or Ra-Thor-assisted batches.

## 7. Future Evolution

- Full per-NPC calls to `geometric-intelligence` engine
- NPC organizations and collective harmony fields
- Player-created NPCs or AI companions
- Deep integration with player housing and personal spaces

---

**This system forms the living heart of Powrush.**

When NPCs feel real, the entire world feels real.