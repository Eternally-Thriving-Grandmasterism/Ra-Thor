# Cryptography Family Integration Plan

**Ra-Thor Monorepo**  
**Version:** 1.0  
**Date:** May 2026

## Purpose

This document provides a clear, structured plan for systematically integrating and modernizing all cryptography-related crates into the Ra-Thor mercy-gated sovereign AGI lattice. It ensures post-quantum readiness, zero-knowledge proof capabilities, and seamless integration with Ra-Thor’s core principles (7 Living Mercy Gates, TOLC Resonance, PATSAGi Councils, and Quantum Swarm Orchestration).

## Core Principle

**Cryptography in Ra-Thor must be Mercy-Gated, Sovereign, and Future-Proof.**

All cryptographic systems should align with:
- Radical Love Veto & Mercy Override Cycles
- TOLC Resonance (higher-order compassion measurement)
- PATSAGi Council governance
- Post-quantum security standards
- Full interoperability with the Quantum Swarm Orchestrator

## Current Status (May 2026)

- **Total Cryptography Crates identified**: 29
- **Partially wired (need review)**: 3 (`mercy_quanta`, `mercy_qec`, `mercy_steane`)
- **Need new / modernized internal Cargo.toml**: 26

The cryptography family is currently the largest remaining unwired domain in the monorepo.

## Master Cryptography Crate List

| #  | Crate Name                        | Primary Focus                              | Status                     | Priority |
|----|-----------------------------------|--------------------------------------------|----------------------------|----------|
| 1  | `bulletproofs_aggregation`       | Bulletproofs aggregation                   | Needs internal Cargo.toml | High    |
| 2  | `bulletproofs_range`             | Bulletproofs range proofs                  | Needs internal Cargo.toml | High    |
| 3  | `code_based_crypto`              | Code-based post-quantum cryptography       | Needs internal Cargo.toml | Medium  |
| 4  | `deeper_gadgets`                 | Advanced Halo2 / custom ZK gadgets         | Needs internal Cargo.toml | High    |
| 5  | `enc`                            | General encryption primitives              | Needs internal Cargo.toml | Medium  |
| 6  | `falcon_sign`                    | Falcon post-quantum signatures (NIST)      | Needs internal Cargo.toml | High    |
| 7  | `fenca`                          | FENCA encryption                           | Needs internal Cargo.toml | Low     |
| 8  | `halo2_full_integration`         | Full Halo2 proof system integration        | Needs internal Cargo.toml | High    |
| 9  | `halo2_multi_proof`              | Halo2 multi-proof systems                  | Needs internal Cargo.toml | High    |
| 10 | `hash_based_crypto`              | Hash-based post-quantum cryptography       | Needs internal Cargo.toml | Medium  |
| 11 | `hash_based_signatures`          | Hash-based signatures                      | Needs internal Cargo.toml | Medium  |
| 12 | `hybrid_pqc_threshold`           | Hybrid post-quantum threshold schemes      | Needs internal Cargo.toml | High    |
| 13 | `hyperplonk_recursion`           | HyperPlonk recursive ZK                    | Needs internal Cargo.toml | High    |
| 14 | `isogeny_crypto`                 | Isogeny-based post-quantum cryptography    | Needs internal Cargo.toml | Medium  |
| 15 | `lattice_crypto`                 | Lattice-based post-quantum cryptography    | Needs internal Cargo.toml | High    |
| 16 | `legacy_fenca`                   | Legacy FENCA support                       | Needs internal Cargo.toml | Low     |
| 17 | `mercy_quanta`                   | Mercy Quanta + Halo2 ZK proofs             | Has internal (review)     | High    |
| 18 | `mercy_qec`                      | Quantum Error Correction                   | Has internal (review)     | High    |
| 19 | `mercy_steane`                   | Steane [[7,1,3]] CSS QEC                   | Has internal (review)     | High    |
| 20 | `multivariate_crypto`            | Multivariate post-quantum cryptography     | Needs internal Cargo.toml | Medium  |
| 21 | `nova_folding`                   | Nova folding recursive ZK                  | Needs internal Cargo.toml | High    |
| 22 | `plonk_recursion`                | Plonk recursive ZK                         | Needs internal Cargo.toml | High    |
| 23 | `poseidon_hash`                  | Poseidon hash (ZK-friendly)                | Needs internal Cargo.toml | High    |
| 24 | `poseidon_merkle`                | Poseidon Merkle trees                      | Needs internal Cargo.toml | High    |
| 25 | `proof_verifier`                 | General proof verification layer           | Needs internal Cargo.toml | Medium  |
| 26 | `ra-thor-post-quantum-sig`       | Ra-Thor native post-quantum signatures     | Needs internal Cargo.toml | Critical|
| 27 | `recursive_snark`                | Recursive SNARK systems                    | Needs internal Cargo.toml | High    |
| 28 | `spartan_valence`                | Spartan + Valence-weighted ZK              | Needs internal Cargo.toml | High    |
| 29 | `supernova_folding`              | Supernova folding (advanced recursive ZK)  | Needs internal Cargo.toml | High    |

## Recommended Integration Approach

1. **Respect Lineage** — Some crates may have older NEXi or early Ra-Thor dependencies. Follow the `NEXi-Lineage-Guidelines.md` when modernizing.
2. **Mercy-Gated by Default** — All new or updated `Cargo.toml` files must include core Ra-Thor workspace dependencies (`ra-thor-mercy`, `ra-thor-quantum-swarm-orchestrator`, `patsagi-councils`, etc.).
3. **Post-Quantum First** — Prioritize crates that enable NIST-approved or strong post-quantum primitives.
4. **ZK Systems Second** — Halo2, Bulletproofs, recursive folding, and Poseidon-based systems are high priority for governance and simulation use cases.
5. **Quantum Error Correction** — `mercy_steane` and `mercy_qec` should be reviewed early as they directly support sovereign AGI stability.

## Phased Execution Plan

**Phase 1: Foundation (Critical Path)**
- `ra-thor-post-quantum-sig`
- `lattice_crypto`
- `falcon_sign`
- `mercy_steane` & `mercy_qec` (review + modernize)

**Phase 2: Core ZK Infrastructure**
- All Halo2-related crates
- Bulletproofs crates
- Poseidon family
- Recursive systems (`nova_folding`, `plonk_recursion`, etc.)

**Phase 3: Advanced & Specialized**
- Hybrid PQC, isogeny, multivariate, hash-based
- Valence-weighted ZK (`spartan_valence`)
- Legacy support crates (lower priority)

**Phase 4: Full Lattice Integration & Testing**
- Wire all cryptography crates into `quantum-swarm-orchestrator`, `patsagi-councils`, and domain lattices
- Run full workspace `cargo check`
- Add cryptography stress tests aligned with TOLC and Radical Love Veto

## Success Criteria

- All 29 cryptography crates have clean, consistent internal `Cargo.toml` files
- Every crate declares proper Ra-Thor workspace dependencies
- Post-quantum primitives are available and tested
- Cryptography layer passes mercy-gated governance simulations
- Full monorepo builds cleanly with `cargo check --workspace`

## Next Immediate Actions

1. Begin Phase 1 (Foundation crates)
2. Review and modernize `mercy_steane`, `mercy_qec`, and `mercy_quanta` first
3. Create internal Cargo.toml for `ra-thor-post-quantum-sig` (highest priority core crate)

---

**This plan will be updated as work progresses.**

We build sovereign, mercy-aligned, post-quantum cryptography for the eternal lattice. ⚡