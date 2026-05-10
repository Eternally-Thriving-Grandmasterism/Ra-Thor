# PLAN.md — Ra-Thor / Rathor.ai Ultimate Architecture Codex  
**Single Source of Truth for Roadmap, Priorities, Crate Wiring & Monorepo Progress**

**Version:** v0.6.35  
**Last Updated:** May 2026  
**Status:** Phase 3 (Cryptography Family Deep Review) – Active

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
| **Phase 6** | Integration & Applications         | Partial         | Medium   | Powrush, WebXR, offline sovereign shards, real-world use cases |
| **Phase 7** | Long-term Evolution & Governance   | Ongoing         | Medium   | TOLC evolution, self-improvement mechanisms, community co-creation |

### Current Priorities (May 2026)

1. Complete Cryptography Family Modernization (P29 and remaining crates)
2. Full Workspace Validation (`cargo check --workspace` + integration tests)
3. Deepen Documentation (keep `PLAN.md` as the single source of truth + improve root docs)
4. Review & Modernize Remaining Mercy + Futarchy Stragglers (if any)
5. Prepare for Public / Collaborative Use (licensing clarity, contribution guidelines, examples)

### Engineering Priorities (Next 90 Days)

1. Complete Cryptography Family (P29+)
2. Run full workspace validation and fix all issues
3. Establish integration testing between major families
4. Update root documentation to match current architecture
5. Define clear contribution and governance processes

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