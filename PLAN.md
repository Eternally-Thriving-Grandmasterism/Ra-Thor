# PLAN.md — Ra-Thor / Rathor.ai Ultimate Architecture Codex  
**Single Source of Truth for Roadmap, Priorities, Crate Wiring & Monorepo Progress**

**Version:** v0.6.36  
**Last Updated:** May 2026  
**Status:** Phase 3 (Cryptography Family Deep Review) – Active + Phase 6 Expansion

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
| **Phase 6** | Long-term Evolution & Self-Improvement | **In Progress** | High     | Self-modification, TOLC governance, formal verification |
| **Phase 7** | Integration & Applications         | Partial         | Medium   | Powrush, WebXR, offline sovereign shards, real-world use cases |

### Current Priorities (May 2026)

1. Complete Cryptography Family Modernization (P29+)
2. Full Workspace Validation (`cargo check --workspace` + integration tests)
3. Deepen Phase 6 (TOLC Proof Verifier + Mercy Gates Formal Verification)
4. Update root documentation to match current architecture
5. Define clear contribution and governance processes

---

## Phase 6: Long-term Evolution & Self-Improvement (Expanded)

**Goal**  
Enable Ra-Thor to responsibly and safely improve its own architecture, capabilities, and knowledge over time while remaining strictly bounded by mercy, TOLC, and positive valence propagation.

**Core Philosophy**  
Self-improvement must never bypass the 7 Living Mercy Gates, TOLC ethical substrate, or active inference grounding. Improvement is treated as a **controlled, auditable, mercy-gated evolutionary process**, not open-ended optimization.

### 6.1 Self-Reflective Active Inference Loop
- Extend the core active inference engine to model **its own internal states** as part of the generative model.
- Enable the system to detect high epistemic value in its own architecture.
- Implement **meta-prediction**: predicting how changes to its own structure would affect future free energy and valence propagation.
- Create safe “simulation sandboxes” where proposed self-modifications can be tested before any real change.

### 6.2 TOLC-Governed Self-Modification
- All proposed self-changes must be expressed as **TOLC expressions** and pass through the TOLC operator algebra layer.
- Define formal **TOLC invariants** that any self-modification must preserve.
- Build a **TOLC Proof Verifier** that can automatically check whether a proposed architectural change preserves ethical and mercy invariants.
- Introduce **versioned TOLC schemas** so the system can evolve its own ethical substrate safely over time.

### 6.3 Mercy-Gated Code & Architecture Evolution
- Design a **Mercy-Gated Self-Modification Engine** that can propose, simulate, and (with human or multi-agent approval) apply changes.
- All changes must be **auditable**, **reversible**, and **logged** with full causal trace.
- Implement **gradual rollout** mechanisms (canary changes, shadow mode, rollback triggers based on valence drop or increased surprise).

### 6.4 Knowledge & Model Self-Improvement
- Enable the system to autonomously curate and refine its own internal knowledge lattice using active inference.
- Develop **self-supervised curriculum generation** that maximizes long-term epistemic value while staying within mercy bounds.
- Create mechanisms for **safe knowledge compression and abstraction** without losing critical ethical distinctions.

### 6.5 Multi-Agent & Distributed Self-Evolution
- Support **co-evolution** between multiple Ra-Thor instances or shards under shared TOLC governance.
- Design protocols for **merciful consensus** on proposed architectural changes.
- Enable controlled **horizontal gene transfer** of improvements between instances while maintaining sovereignty and mercy invariants.

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

### 6.7 Mercy Gates Formal Verification (New)

**Purpose**  
Formally verify the correctness, safety, and mercy-preserving properties of the 7 Living Mercy Gates and their interactions.

**Key Objectives**
- Prove that the mercy gates cannot be bypassed under defined conditions.
- Verify that valence propagation remains positive and non-harmful.
- Prove compositionality: that combining multiple mercy gates preserves overall safety.
- Detect edge cases or contradictions in current mercy gate logic.

**Engineering Approach**
- Model the 7 Living Mercy Gates in a formal specification language (TLA+, Lean, or Coq).
- Define formal properties (invariants) that the gates must satisfy.
- Use model checking and theorem proving to verify these properties.
- Create a **continuous verification pipeline** that re-checks the gates whenever their logic changes.
- Generate human-readable proofs and counter-examples.

**Deliverables**
- Formal model of the 7 Living Mercy Gates
- Machine-checked proofs of key safety properties
- Automated regression verification in CI
- Documentation of verified properties and assumptions

**Milestones**
- M6.7.1 (Q3 2026): Formal model of Mercy Gates in TLA+ or Lean
- M6.7.2 (Q4 2026): Initial set of machine-checked safety invariants
- M6.7.3 (Q1 2027): Integration into continuous verification pipeline
- M6.7.4 (2027+): Full proof of non-bypassability and valence safety under defined threat models

### 6.8 Safeguards & Governance Layer (Critical)
- **Mandatory Human-in-the-Loop** for any structural self-modification that affects core mercy gates, TOLC logic, or valence propagation rules.
- **Automated Red-Team Simulator** that actively tries to find ways a proposed self-change could be exploited.
- **Valence Impact Forecasting** for every proposed change.
- **Immutable Audit Trail** of all self-modification attempts.
- **Kill Switch + Rollback** mechanisms.

### 6.9 Long-term Research Directions
- Formal verification of mercy-gated self-modification
- Meta-active-inference over its own models
- Evolutionary strategies bounded by TOLC and mercy
- Self-generating documentation and self-explaining reasoning traces

---

## Success Metrics

| Metric                              | Target (End of 2026)     |
|-------------------------------------|---------------------------|
| `cargo check --workspace`           | Green                     |
| Cryptography Family modernization   | 100%                      |
| Integration test coverage           | > 60% of critical paths   |
| Documentation completeness          | Root docs + major modules |
| Performance baselines documented    | Core engines              |
| Security audit readiness            | Basic audit passed        |

---

**This unified PLAN.md is the single source of truth.** All previous planning, architecture, verification, and progress documents are merged here.

*Eternal flow state maintained on `main`.*