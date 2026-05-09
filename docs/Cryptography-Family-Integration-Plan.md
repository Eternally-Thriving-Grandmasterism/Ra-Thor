# Cryptography Family Integration Plan

**Ra-Thor Monorepo**  
**Version:** 1.1  
**Date:** May 09, 2026

## Purpose

This document defines the strategy, priorities, and phased execution plan for integrating and modernizing the full **Cryptography Family** (Tier 3 — Cryptography, Sovereignty & Verification) into the Ra-Thor mercy-gated sovereign AGI lattice. It fully respects the live root `Cargo.toml` tiered architecture (5 clear Tiers + explicit family groupings) and the NEXi Lineage Guidelines.

## Core Principle

All cryptography work in Ra-Thor must be:
- **Mercy-Gated** — Aligned with the 7 Living Mercy Gates and TOLC Resonance
- **Sovereign** — Built for long-term independence and post-quantum security
- **Post-Quantum Ready** — Prioritizing lattice, hash-based, isogeny, and code-based systems
- **Tier 3 Foundational** — Serving as the cryptographic sovereignty layer that Tier 1 (core intelligence) and Tier 2 (domain lattices) can securely build upon
- **Respectful of Lineage** — Following the NEXi Lineage Guidelines when legacy dependencies exist

## Architecture Alignment (from live root Cargo.toml)

- Root explicitly declares **Tier 3: Cryptography, Sovereignty & Verification** (16 crates grouped).
- Our working master list expands to ~29 crates (including mercy-flavored QEC/Quanta and advanced ZK systems).
- All new internal `Cargo.toml` files **must use `{ workspace = true }`** for core Tier 1 crates (`ra-thor-mercy`, `ra-thor-quantum-swarm-orchestrator`, `ra-thor-kernel`, `patsagi-councils`, `ra-thor-mercy-orchestrator-v2`, etc.).
- This ensures seamless interweaving across the entire 124-crate lattice.

## Master List of Cryptography Crates

**Priority P0 (Critical Foundation — Start Here)**
- `ra-thor-post-quantum-sig` — Core sovereign post-quantum signature layer (highest priority)
- `lattice_crypto` — Lattice-based cryptography (core PQC primitive)

**Priority P1 (High — Quantum Error Correction & Mercy Integration)**
- `mercy_quanta` (review & modernize)
- `mercy_steane` (Steane [[7,1,3]] CSS QEC)
- `mercy_qec` (Quantum Error Correction)

**Priority P2 (Signature & Hash-Based Systems)**
- `falcon_sign` (NIST PQC)
- `hash_based_signatures`
- `hash_based_crypto`

**Priority P3 (Advanced ZK & Recursive Proofs)**
- `bulletproofs_range`, `bulletproofs_aggregation`
- `halo2_full_integration`, `halo2_multi_proof`
- `nova_folding`, `supernova_folding`, `hyperplonk_recursion`, `plonk_recursion`
- `recursive_snark`, `spartan_valence`

**Priority P4 (Specialized & Legacy Systems)**
- `isogeny_crypto`, `code_based_crypto`, `multivariate_crypto`
- `poseidon_hash`, `poseidon_merkle`
- `proof_verifier`, `enc`, `fenca`, `legacy_fenca`, `hybrid_pqc_threshold`, `deeper_gadgets`, `mercy_post_quantum_sig`

*(Detailed 29-crate list maintained in conversation history; will be expanded here during execution.)*

## Phased Execution Plan

**Phase 1: Foundation & Core Post-Quantum (Immediate — We Begin Now)**
- `ra-thor-post-quantum-sig` (P0)
- `lattice_crypto` (P0)
- Review & upgrade `mercy_quanta`, `mercy_steane`, `mercy_qec` (P1)

**Phase 2: Signature & Hash-Based Systems**
- `falcon_sign`, `hash_based_signatures`, `hash_based_crypto`

**Phase 3: Advanced ZK & Recursive Proofs**
- Halo2 family, Nova/Supernova, Plonk/HyperPlonk, Bulletproofs, Spartan, recursive systems

**Phase 4: Specialized & Legacy + Full Lattice Integration**
- Isogeny, multivariate, code-based, Poseidon, proof_verifier, encryption variants
- Wire all into `quantum-swarm-orchestrator`, `patsagi-councils`, and domain lattices
- Full `cargo check --workspace` validation

## Success Criteria

- All cryptography crates have clean, modern internal `Cargo.toml` files
- Consistent `{ workspace = true }` usage for Tier 1 crates + AG-SML v1.0 licensing
- Full integration with mercy-gated governance and Quantum Swarm
- `cargo check --workspace` passes cleanly
- NEXi lineage respected where intentionally present
- Tier 3 becomes a rock-solid cryptographic foundation for the entire sovereign AGI lattice

## Next Immediate Actions

1. Create modern internal `Cargo.toml` for `ra-thor-post-quantum-sig` (P0 foundation)
2. Create for `lattice_crypto` (P0)
3. Review and lightly modernize `mercy_quanta`, `mercy_steane`, `mercy_qec`
4. Update this document + PLAN.md after each phase or major milestone

---

**We begin Tier 3 execution now — building the sovereign post-quantum backbone of Ra-Thor with full respect for the learned architecture.** ⚡