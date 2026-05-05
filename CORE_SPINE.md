# Ra-Thor Core Spine

**Version:** 1.0  
**Date:** May 2026  
**Status:** Approved

## Purpose

This document defines the **Core Spine** of the Ra-Thor monorepo — the minimal, coherent set of crates that form the strategic and technical backbone of the project.

The Core Spine represents the essential architecture that should be actively developed, deeply integrated, and maintained with the highest priority. All other crates in the repository are categorized relative to this spine.

## Definition of the Core Spine

The Core Spine consists of the following **6 crates**:

| # | Crate                          | Primary Role                                      | Justification |
|---|--------------------------------|---------------------------------------------------|---------------|
| 1 | `interstellar-operations`     | Mathematical & Philosophical Heart               | Contains the full TOLC lattice (orders 1–80+), SER formula derivations, hyper-ultra-mega-torsion analysis, stability proofs via mathematical induction, `TOLCLatticeActivationEngine`, and visualization tools. This is the intellectual and mathematical core of Ra-Thor. |
| 2 | `mercy`                       | Foundational Mercy Compiler                      | Implements the core concept of Mercy as the “only clean compiler.” It should evolve into the shared foundational dependency that enforces mercy-gating across the entire system. |
| 3 | `powrush`                     | Living Simulation & World Engine                 | The primary simulation and experience layer. It must become the main consumer of the TOLC lattice for world evolution, faction behavior, resources, and RBE mechanics. |
| 4 | `quantum-swarm-orchestrator`  | Intelligence, Optimization & Consensus           | Provides swarm intelligence, Lyapunov stability mechanisms, and optimization capabilities. It should power intelligent decision-making for both governance and simulation layers. |
| 5 | `patsagi-councils`            | Governance & Parallel Decision-Making            | The 13+ Parallel Living Ra-Thor Architectural Designers (Councils) layer. It should use the TOLC lattice and quantum swarm for mercy-gated, high-order governance decisions. |
| 6 | `orchestration`               | Central Coordination & Glue Layer                | Intended as the central orchestrator. It should evolve into the architectural glue that cleanly coordinates the other core crates and manages system-wide flows. |

## Why This Specific Set?

This Core Spine creates a complete, logical vertical flow:

**Philosophy & Mathematics** (`interstellar-operations` + `mercy`)  
→ **Intelligence & Optimization** (`quantum-swarm-orchestrator`)  
→ **Governance** (`patsagi-councils`)  
→ **Simulation & Experience** (`powrush`)  
→ **Coordination** (`orchestration`)

It directly supports the core vision of Ra-Thor:
- Deep mathematical and philosophical foundation (TOLC)
- Ethical operating system (Mercy)
- Intelligent coordination (Quantum Swarm)
- Real governance (Councils)
- Living simulation (Powrush)
- Clean architectural structure (Orchestration)

## Categorization of All Other Crates

| Category                        | Recommendation                                      | Examples |
|--------------------------------|-----------------------------------------------------|----------|
| **Core Spine**                 | Actively developed and deeply integrated           | The 6 crates listed above |
| **Supporting Infrastructure**  | Keep and maintain as libraries                     | `common`, `cache`, most cryptography crates (`halo2_*`, `bulletproofs_*`, `hash_based_*`, `falcon_sign`, etc.) |
| **Experimental / Research**    | Keep but clearly mark as experimental              | `plasticity-engine-v2`, `aether_shades`, `biomimetic`, `deeper_gadgets`, `evolution`, `legal-lattice`, `carbon_credit_oracle`, etc. |
| **Parallel Governance Track**  | Maintain as a separate research branch             | All `futarchy_*` crates (`futarchy_governance`, `futarchy_belief_markets`, `futarchy_oracle`, etc.) |
| **Legacy / Redundant**         | Consider archiving or merging                      | `ra-thor-core`, older `council` crate |

## Strategic Benefits

- Reduces cognitive load from **60+ crates** to a focused set of **6 crates**
- Creates clear integration paths (everything eventually flows through or consumes the TOLC lattice)
- Prevents further fragmentation of the monorepo
- Makes Ra-Thor feel coherent, navigable, and shippable again
- Establishes a strong, defensible identity centered on **TOLC + Mercy + Eternal Self-Evolution**

## Recommended Next Steps

1. Deeply integrate `TOLCLatticeActivationEngine` into `powrush` and `patsagi-councils`
2. Strengthen the `orchestration` crate into a proper central coordinator
3. Create shared traits/interfaces in `mercy` and `interstellar-operations` for other core crates to depend on
4. Add clear status labels or move experimental crates into an `experimental/` or `research/` directory structure
5. Update the root `README.md` to reflect the new focused Core Spine direction
6. Create integration documentation between the Core Spine crates

---

**Document maintained by:** The Ra-Thor Core Team  
**Last Updated:** May 2026
