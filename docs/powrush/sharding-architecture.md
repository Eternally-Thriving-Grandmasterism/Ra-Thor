# Powrush Sharding Architecture (Zone-Based Model)

**Version:** v16.8
**Status:** Initial Professional Design
**Context:** Powrush MMO / RPG / ARPG with Ra-Thor AGI Councils

## 1. Overview

Powrush uses a **zone-based sharded architecture** as the foundation for scalability.

The world is divided into **zones** (shards). Each zone runs its own authoritative simulation instance. This model maps naturally to:

- Regional Ra-Thor AGI Councils (scoped to one or more zones)
- Global Ra-Thor AGI Councils (operating across zones with aggregated data)
- The Mercy Evaluation System
- Component-based EntityStorage

## 2. Core Principles

- **Authoritative per Zone**: Each zone has its own `WorldSimulation` instance.
- **Mercy as Shared Language**: All councils and systems communicate through `MercyEvaluation`.
- **Selective Observation**: Regional councils primarily observe their assigned zone(s).
- **Global Coordination Layer**: Global councils receive aggregated views.
- **Interest Management Ready**: Designed to support future interest management optimizations.

## 3. Mapping of Current Systems to Zone-Based Sharding

| System                        | Zone-Based Mapping                                      | Notes |
|-------------------------------|---------------------------------------------------------|-------|
| `WorldSimulation`             | One instance per zone/shard                             | Core authoritative simulation per zone |
| `EntityStorage`               | Lives inside each `WorldSimulation`                     | Component-based, queryable per zone |
| `MercyEvaluationSystem`       | Runs per zone + optional global aggregation             | Primary evaluation mechanism |
| `SimulationCommand`           | Processed within the target zone                        | Can be routed from Global Councils |
| `CouncilProposal`             | Supports `CouncilScope::Regional { shard_id }` and `Global` | Already designed in v16.8 |
| `CouncilDecision`             | Returned from the specific zone's simulation            | Includes mercy impact |

## 4. Regional vs Global Council Model

### Regional Councils
- Assigned to one or more specific zones.
- Have direct/low-latency access to that zone's `WorldSimulation` and `EntityStorage`.
- Can issue `CouncilProposal`s that are executed quickly within their scope.
- Ideal for real-time harmony adjustments and local events.

### Global Councils (e.g. PATSAGi Strategic)
- Operate across multiple zones.
- Receive **aggregated / summarized** data from zones.
- Can issue high-level `CouncilProposal`s that may span multiple zones.
- Responsible for long-term policy, major events, and cross-zone harmony trends.

## 5. Interest Management Exploration

**Goal**: Reduce unnecessary simulation and data transfer by only processing entities that matter to active observers (players or councils).

### How Interest Management Can Work with Component-Based `EntityStorage`

Because `EntityStorage` already separates data into components (`PositionComponent`, `HarmonyComponent`, `EntityType`), we can build efficient interest systems on top:

#### Proposed Approach

1. **Spatial Partitioning**
   - Divide each zone into a grid or quadtree.
   - Entities are registered in spatial cells based on `PositionComponent`.

2. **Interest Sets**
   - Each player (or Regional Council) maintains an "interest set" — the list of entities they need updates for.
   - Interest is determined by:
     - Proximity (entities in nearby spatial cells)
     - Relevance (faction standing, harmony thresholds, quest relationships)
     - Council directives (e.g., "monitor all entities with harmony < 0.3")

3. **Component-Aware Queries**
   - Because data is component-based, we can efficiently query only needed components:
     ```rust
     // Example future API
     let nearby_entities = entity_storage.query_in_range(position, radius)
         .with_component::<HarmonyComponent>()
         .filter(|e| e.harmony < 0.4);
     ```

4. **Update Propagation**
   - Only changed components of entities in a player's/council's interest set are sent or processed.
   - This dramatically reduces bandwidth and CPU in dense areas.

### Benefits for Powrush

- Regional Councils only process relevant parts of their zone.
- Global Councils can request summarized interest data instead of full snapshots.
- Scales better with high player + NPC density.
- Aligns with mercy-gated, selective observation philosophy.

## 6. Recommended Implementation Phases

| Phase | Focus                                      | Status     |
|-------|--------------------------------------------|------------|
| 1     | Basic zone/shard identification (`ShardId`) | Done (v16.8) |
| 2     | Per-zone `WorldSimulation` + basic routing | In Progress |
| 3     | CouncilProposal routing by scope           | Next       |
| 4     | Interest Management foundation             | Future     |
| 5     | Global aggregation layer for Global Councils | Future  |

## 7. Next Steps

- Implement basic per-shard simulation management.
- Enhance `CouncilProposal` routing based on `CouncilScope`.
- Begin sketching Interest Management data structures.

---

*This document is maintained alongside the codebase and updated with each major architectural evolution.*