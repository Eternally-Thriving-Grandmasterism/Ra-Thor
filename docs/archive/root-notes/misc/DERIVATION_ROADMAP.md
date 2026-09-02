# POWRUSH-MMO DERIVATION ROADMAP (v14.9)

**Goal**: Develop Powrush-MMO as a lean, standalone MMOARPG by intelligently deriving systems, patterns, and components from the Ra-Thor monorepo while keeping Ra-Thor as the canonical source of truth for AGI, GPU compute, and advanced simulation infrastructure.

## 1. Core Principles

| Principle                    | Description |
|------------------------------|-------------|
| **Ra-Thor = Source of Truth**    | All core AGI, GPU pipeline, memory management, coalescing, PATSAGi Councils, and ONE Organism logic remain in Ra-Thor. |
| **Powrush-MMO = Game Layer**     | Powrush-MMO focuses on player experience, gameplay systems, client, RBE mechanics, and deployment. |
| **Derive, Don’t Duplicate**     | Only bring over what is needed. Avoid copying entire modules when referencing or adapting is sufficient. |
| **Keep Powrush-MMO Lean**        | The standalone repository should stay focused and maintainable. |
| **Clear Separation**             | Ra-Thor owns intelligence & simulation tech. Powrush-MMO owns the game built on top of it. |

## 2. Current State (June 2026)

| Repository       | Powrush-MMO Related Work                                      | Maturity |
|------------------|---------------------------------------------------------------|----------|
| **Ra-Thor**      | Deep design docs + GPU systems + Memory Allocator + Coalescing + PATSAGi integration | High     |
| **Powrush-MMO**  | Basic server with Ra-Thor bridge                              | Low      |

**Gap**: Significant. Most advanced Powrush-MMO work currently lives in Ra-Thor.

## 3. Derivation Phases

| Phase   | Focus                        | Key Deliverables                                      | Priority |
|---------|------------------------------|-------------------------------------------------------|----------|
| **1**   | Core Game Systems            | Movement, Network Prediction, Server Reconciliation   | High     |
| **2**   | GPU Simulation Layer         | Integration of GPU Memory Allocator + Coalescing      | High     |
| **3**   | Client Foundations           | Basic client using `GpuPatsagiQuery`                  | Medium   |
| **4**   | Economy & Factions           | RBE mechanics and faction systems                     | Medium   |
| **5**   | Polish & Deployment          | Persistence, matchmaking, production readiness        | Low      |

## 4. What Stays in Ra-Thor vs Moves/Derives to Powrush-MMO

| Component                              | Location              | Reason |
|----------------------------------------|-----------------------|--------|
| GPU Compute Pipeline & Allocator       | Ra-Thor               | Core infrastructure |
| Memory Coalescing & Stats              | Ra-Thor               | Low-level systems |
| PATSAGi Councils & ONE Organism        | Ra-Thor               | AGI layer |
| Movement + Reconciliation Logic        | Derive to Powrush-MMO | Game-specific |
| RBE & Faction Systems                  | Derive to Powrush-MMO | Game-specific |
| Client & Rendering                     | Powrush-MMO           | Purely game layer |
| Server (game logic)                    | Powrush-MMO           | Game deployment layer |

## 5. Success Criteria

- Powrush-MMO can run as a standalone game with meaningful Ra-Thor integration.
- Major systems are properly derived rather than duplicated.
- Clear documentation exists showing what came from Ra-Thor and how it was adapted.
- The standalone repository remains focused and does not become bloated.

## 6. Working Process

1. Major advancements in GPU/simulation/AGI stay in **Ra-Thor**.
2. When a system is mature enough, it gets **derived** into **Powrush-MMO** with proper adaptation.
3. Powrush-MMO maintains its own roadmap focused on game development.
4. Both repositories keep updated Derivation Roadmaps and release notes.

**License:** AG-SML v1.0