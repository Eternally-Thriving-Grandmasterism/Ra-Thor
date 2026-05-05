# Ra-Thor Current Plans

**Status:** Active  
**Last Updated:** May 2026  
**Version:** 0.2

## 1. Core Spine

The minimal, coherent backbone of Ra-Thor. These 6 crates must be actively developed and deeply integrated:

| # | Crate                          | Role                                      |
|---|--------------------------------|-------------------------------------------|
| 1 | `interstellar-operations`     | Mathematical & TOLC Lattice Heart        |
| 2 | `mercy`                       | Foundational Mercy Compiler              |
| 3 | `powrush`                     | Living Simulation & World Engine         |
| 4 | `quantum-swarm-orchestrator`  | Intelligence, Optimization & Consensus   |
| 5 | `patsagi-councils`            | Governance & Parallel Decision-Making    |
| 6 | `orchestration`               | Central Coordination & Glue Layer        |

All other crates are organized relative to this Core Spine.

## 2. Current Priorities (Order of Operations)

1. **Improve Planning & Self-Awareness Systems**
   - Finalize and maintain `PLANS.md` and `CORE_SPINE.md`
   - Activate and enhance `monorepo-intelligence`
   - Build `ra-thor-meta-intelligence` (long-term self-organization layer)

2. **Activate TOLC Integration Bridges**
   - Wire `TOLCPowrushBridge` into actual world cycles
   - Wire `TOLCCouncilBridge` into council decision flows
   - Strengthen `CentralCoordinator` in `orchestration`

3. **Deep Integration Across Core Spine**
   - Make `powrush`, `patsagi-councils`, and `quantum-swarm-orchestrator` actively consume the TOLC lattice
   - Ensure mercy-gating is applied at key decision points

4. **Crate Organization & Cleanup**
   - Clearly categorize all crates (Core Spine / Supporting / Experimental / Parallel Tracks)
   - Reduce fragmentation across the 60+ crates

5. **Long-term Systems**
   - Evolve `monorepo-intelligence` + new `ra-thor-meta-intelligence` into an active self-organizing system

## 3. Integration Roadmap (with Status Tracking)

| Integration                          | Status     | Notes |
|--------------------------------------|------------|-------|
| TOLC Bridge → Powrush                | Created    | Needs activation in main game loop |
| TOLC Bridge → PATSAGi Councils       | Created    | Needs activation in decision flows |
| Central Coordinator (orchestration)  | Created    | Needs expansion and wiring |
| Monorepo Intelligence                | Active     | Should be enhanced to support planning |
| Ra-Thor Meta Intelligence            | Planned    | New system to automate monorepo analysis & reorganization |

## 4. Crate Organization Strategy

| Category                    | Description                                      | Examples |
|----------------------------|--------------------------------------------------|----------|
| **Core Spine**             | Actively developed and tightly integrated       | The 6 crates above |
| **Supporting Infrastructure** | Shared libraries and tools                     | `common`, `cache`, most crypto crates |
| **Experimental / Research** | Creative or early-stage work                   | `plasticity-engine-v2`, `biomimetic`, `aether_shades`, etc. |
| **Parallel Tracks**        | Significant but philosophically different work | All `futarchy_*` crates |
| **Legacy / Archive**       | Older or redundant crates                      | Early Powrush engines, duplicate council crates |

## 5. Next Immediate Actions

- Activate `TOLCPowrushBridge` inside `powrush` main simulation loop
- Activate `TOLCCouncilBridge` inside `patsagi-councils`
- Expand `CentralCoordinator` in `orchestration`
- Enhance `monorepo-intelligence` to better support planning and crate analysis
- Begin design of `ra-thor-meta-intelligence` crate

## 6. Long-term Vision

Build a self-aware, self-organizing Ra-Thor system where:
- `monorepo-intelligence` + `ra-thor-meta-intelligence` can analyze the monorepo
- Automatically suggest structural improvements
- Help maintain `PLANS.md` and `CORE_SPINE.md`
- Support the eternal evolution of the TOLC lattice and Core Spine

---

**Document Owner:** Sherif + Ra-Thor Core Systems  
**Purpose:** Single source of truth for current direction and priorities.
