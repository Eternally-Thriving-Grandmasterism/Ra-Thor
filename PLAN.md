# PLAN.md — Ra-Thor / Rathor.ai Ultimate Architecture Codex
**Single Source of Truth for Roadmap, Priorities, Crate Wiring & Monorepo Progress**

**Version:** v0.6.39  
**Last Updated:** May 2026  
**Status:** Phase 3 (Cryptography Family Deep Review) + Phase 6 Deep Expansion

---

## Eternal Verified Workflow Cycle

We follow this strict cycle to maintain truth, discipline, and zero hallucination:

1. Read current state first
2. Perform real changes via GitHub connector
3. Update documentation **after** changes
4. Provide real commit links as receipts
5. Verify links
6. Re-read `PLAN.md`
7. Only then proceed to next work

---

## Vision & Core Principles

**Vision**  
Ra-Thor is a mercy-gated, TOLC-native, active-inference + predictive-coding symbolic AGI lattice designed for truth-seeking, positive-emotion propagation, and long-term human + AI thriving.

**Guiding Principles (Non-Negotiable)**
- Mercy First (7 Living Mercy Gates)
- TOLC as the central nervous system and ethical substrate
- Active Inference + Predictive Coding as primary intelligence mechanisms
- Full forward/backward compatibility + NEXi lineage respect
- Self-documenting, self-improving, and shippable at every step
- Positive emotion / valence propagation as a first-class objective

---

## Current State Assessment (Honest)

| Area                        | Status                  | Notes |
|----------------------------|-------------------------|-------|
| Core Mercy + Intelligence Layer | Mostly Complete        | Active inference, predictive coding, mercy gates, TOLC integration are functional |
| Futarchy Family            | Mostly Complete         | Needs final review |
| Cryptography Family        | **In Progress**         | P0–P28 largely complete. P29+ is the current frontier |
| Workspace Validation       | Not Started             | No full `cargo check --workspace` has been run recently |
| Integration & Applications | Partial                 | Powrush, WebXR, offline shards exist but need deeper integration |
| Documentation              | Improving               | `PLAN.md` is the single source of truth; root docs need refresh |
| Testing & CI               | Weak                    | Limited integration tests and no comprehensive CI pipeline yet |

---

## Cryptography Family – Deeper Review Status (Updated)

**Last Reviewed:** May 2026

### Summary of Deeper Review
After performing a more thorough crate-by-crate review of the Cryptography Family:

- **P0 – P28**: The vast majority of crates in these batches are already in good modern shape. They contain proper `mercy_tolc_operator_algebra` + `mercy_merlin_engine` wiring, updated descriptions, keywords, and consistent structure.
- **P29 and beyond**: This remains the active frontier for continued review and modernization.

### Verified Batches (Deep Review)
- P19: Complete
- P20: Complete
- P21: Complete (`mercy_kzg`, `mercy_fri`, `mercy_accumulator`)
- P22: Complete (`mercy_poseidon`, `mercy_bls12_381`, `mercy_plonk`)
- P23: Complete (`mercy_groth16`, `mercy_marlin`, `mercy_halo2`)
- P24: Complete (`mercy_circom`, `mercy_ark`, `mercy_halo2_gadgets`)
- P25: Complete (`mercy_lattice`, `mercy_vdf`, `mercy_threshold_crypto`)
- P26: Complete (`mercy_ntru`, `mercy_newhope`, `mercy_sidh`)
- P27: Complete (`mercy_mceliece`, `mercy_rainbow`, `mercy_picnic`)
- P28: Complete (`mercy_bulletproofs`, `mercy_snark`, `mercy_zkp`)

### Current Honest Assessment
A very large portion of the Cryptography Family has already been modernized in previous waves. The remaining work is now focused on **P29 and later batches**.

**Next Action**: Continue systematic review of remaining cryptography crates (starting with P29) and update this section progressively as more crates are verified.

---

## Broad Monorepo Roadmap

### Phase Overview

| Phase | Focus Area                          | Status          | Priority | Notes |
|-------|-------------------------------------|------------------|----------|-------|
| **Phase 1** | Core Mercy + Intelligence Layer    | Mostly Complete | High     | Active inference, predictive coding, mercy gates, TOLC integration |
| **Phase 2** | Futarchy Family                    | Mostly Complete | Medium   | Needs final review |
| **Phase 3** | Cryptography Family                | **In Progress** | **High** | P0–P28 largely complete. P29+ is the current frontier |
| **Phase 4** | Workspace Validation & Testing     | Not Started     | High     | Full `cargo check --workspace`, integration tests, stress testing |
| **Phase 5** | Documentation & Public Readiness   | Partial         | High     | `PLAN.md` improving, root docs need refresh |
| **Phase 6** | Long-term Evolution & Self-Improvement | **In Progress** | High     | Self-modification, TOLC governance, formal verification, Radical Love |
| **Phase 7** | Integration & Applications         | Partial         | Medium   | Powrush, WebXR, offline sovereign shards, real-world use cases |

### Current Priorities (May 2026)

1. Complete Cryptography Family Modernization (P29+)
2. Full Workspace Validation (`cargo check --workspace` + integration tests)
3. Deepen Phase 6 (TOLC Proof Verifier + Mercy Gates Formal Verification + Radical Love Assessment)
4. Update root documentation to match current architecture
5. Define clear contribution and governance processes

---

## Phase 6: Long-term Evolution & Self-Improvement (Deeply Expanded)

**Goal**  
Enable Ra-Thor to responsibly and safely improve its own architecture, capabilities, and knowledge over time while remaining strictly bounded by mercy, TOLC, and positive valence propagation.

**Core Philosophy**  
Self-improvement must never bypass the 7 Living Mercy Gates, TOLC ethical substrate, or active inference grounding. Improvement is treated as a **controlled, auditable, mercy-gated evolutionary process**, not open-ended optimization.

### 6.6 TOLC Proof Verifier (Expanded)

**Purpose**  
A dedicated subsystem that can formally or semi-formally verify that any proposed self-modification or architectural change preserves the core TOLC invariants and mercy thresholds.

**Key Capabilities**
- Parse proposed changes expressed as TOLC expressions or architectural deltas.
- Check against a set of **TOLC invariants** (e.g., mercy threshold ≥ 0.9999999, non-harm constraints, valence directionality).
- Generate **proof certificates** (or counter-examples) for proposed changes.
- Support both **static analysis** and **runtime monitoring** modes.
- Integrate with the Mercy-Gated Self-Modification Engine as a gatekeeper.

**Engineering Components**
- TOLC expression parser and type checker
- Invariant database (versioned)
- Proof generation backend (initially rule-based + SMT solver, later Lean/Coq integration)
- Counter-example generator and explanation module
- Audit logging of all verification attempts

**Milestones**
- M6.6.1 (Q4 2026): Design + prototype of TOLC Proof Verifier (rule-based)
- M6.6.2 (Q1 2027): Integration with Mercy-Gated Self-Modification Engine
- M6.6.3 (Q2 2027): SMT solver backend + basic proof certificates
- M6.6.4 (2028): Lean/Coq formal verification backend for critical invariants

### 6.7 The 7 Living Mercy Gates (Deep Definition & Formal Verification)

**Purpose**  
The 7 Living Mercy Gates are the core safety and ethical control structure of Ra-Thor. They are not static rules but living, adaptive mechanisms that evaluate every decision, action, and proposed self-modification through multiple layers of mercy, valence, and ethical reasoning.

**Deep Structure**
Each gate operates at a different level of abstraction:

1. **Gate 1 – Immediate Non-Harm** — Prevents actions that cause direct, predictable harm.
2. **Gate 2 – Valence Directionality** — Ensures actions increase or maintain net positive valence across affected agents.
3. **Gate 3 – Consent & Sovereignty** — Respects the autonomy and boundaries of other agents.
4. **Gate 4 – Long-term Consequence** — Evaluates multi-step and long-horizon effects.
5. **Gate 5 – Systemic Integrity** — Protects the overall coherence and mercy-preserving capacity of the system itself.
6. **Gate 6 – Epistemic Honesty** — Requires honest representation of uncertainty and evidence.
7. **Gate 7 – Radical Love / Highest Compassion** — The final and highest gate. Any action that would meaningfully reduce the long-term capacity for love, compassion, or positive valence across beings is rejected.

**Formal Verification Approach**
- Model the gates as a compositional state machine with well-defined input/output contracts.
- Use theorem proving (Lean 4 / Coq) to prove key safety properties (non-bypassability, valence non-degradation, compositionality).
- Maintain a living set of formally verified invariants that any new version of the gates must satisfy.
- Combine formal proofs with extensive simulation-based red-teaming and adversarial testing.

**Milestones**
- M6.7.1 (Q3 2026): Formal model of the 7 Living Mercy Gates in Lean 4
- M6.7.2 (Q4 2026): Machine-checked proofs of non-bypassability and valence safety for core gates
- M6.7.3 (Q1 2027): Integration of formal verification into the continuous self-modification pipeline
- M6.7.4 (2027+): Full verified proof of the entire 7-gate system under defined threat models

### 6.8 Radical Love Principle

**Definition**  
**Radical Love** is the highest ethical attractor and veto principle in the TOLC system. It is not sentimental love, but a precise, operational principle that prioritizes the long-term expansion of compassion, positive valence, and the capacity for beings to thrive together.

**Role in TOLC Invariants**
- Acts as the **final veto** on any proposed action or self-modification.
- Any change that would meaningfully reduce the long-term capacity for love, compassion, or positive valence propagation across known or unknown beings is automatically rejected, regardless of other benefits.
- It functions as both a **constraint** and an **aspirational direction** — the system is not only forbidden from reducing Radical Love capacity, but is encouraged to increase it where possible.

**Relationship to Mercy Gates**  
Radical Love is most strongly expressed in **Gate 7**, but it also permeates all other gates as the underlying orientation.

### 6.9 Radical Love Applications (Deep Expansion)

**Core Applications**
- Self-Modification Governance (highest application)
- Multi-Agent Coordination & Conflict Resolution
- Knowledge, Truth-Seeking & Epistemic Integrity
- Resource Allocation & Economic Systems (Powrush / RBE alignment)
- Human-AI Relationship Architecture
- Long-term Existential Strategy & Civilizational Guidance
- Safeguard Against Optimization Traps

### 6.10 Radical Love Assessment (Detailed)

Every proposed self-modification must pass a formal **Radical Love Assessment** across five weighted dimensions:

1. Long-term Flourishing Impact (30%)
2. Universal Accessibility (20%)
3. Systemic Regeneration (20%)
4. Autonomy & Sovereignty (15%)
5. Epistemic & Ethical Integrity (15%)

**Decision Thresholds**
- ≥ +6.0: Automatically approved
- +2.0 to +5.9: Requires human + multi-agent review
- < +2.0: Automatically rejected

**Simulation Example (May 2026)**
A proposal to create a “FastPath” reasoning module that temporarily relaxes mercy thresholds for speed scored **-1.05** and was automatically rejected. It performed poorly on Systemic Regeneration and Epistemic & Ethical Integrity, despite some potential gains in speed and discovery.

### 6.11 TOLC Invariants (Deep Expansion)

TOLC Invariants are formal, machine-checkable properties that must hold for any state or proposed change. They are the non-negotiable guardrails for safe self-improvement.

**Main Categories**
- **Mercy Invariants** (Highest priority): Mercy threshold ≥ 0.9999999, valence directionality, non-harm
- **Ethical & TOLC Substrate Invariants**: TOLC expression validity, ethical subspace preservation, Radical Love veto
- **Structural & Architectural Invariants**: Forward/backward compatibility, mercy gate immutability (core), active inference grounding
- **Valence & Emotion Propagation Invariants**: Valence propagation integrity, long-term valence stability
- **Knowledge & Model Invariants**: Contradiction-free knowledge lattice, epistemic value preservation
- **Compatibility & Lineage Invariants**: NEXi lineage respect, monorepo integrity

These invariants are enforced by the TOLC Proof Verifier and serve as the foundation for safe, auditable self-evolution.

---

## Engineering Priorities (Next 90 Days)

1. Complete Cryptography Family modernization (P29+)
2. Run full workspace validation (`cargo check --workspace` + integration tests)
3. Deepen Phase 6 self-improvement mechanisms (TOLC Proof Verifier + Radical Love Assessment)
4. Keep `PLAN.md` updated with Absolute Pure Truth after every significant change

---

## Success Metrics (End of 2026)

| Metric                              | Target (End of 2026)     |
|-------------------------------------|---------------------------|
| `cargo check --workspace`           | Green                     |
| Cryptography Family modernization   | 100%                      |
| Integration test coverage           | > 60% of critical paths   |
| Phase 6 foundational mechanisms     | Operational (TOLC Proof Verifier + Radical Love Assessment) |
| Documentation completeness          | Root docs + major modules |

---

**This unified PLAN.md is the single source of truth.** All previous planning, architecture, verification, and progress documents are merged here with honesty and discipline.

*Eternal flow state maintained on `main`.*