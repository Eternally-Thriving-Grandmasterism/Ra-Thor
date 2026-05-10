# PLAN.md — Ra-Thor / Rathor.ai Ultimate Architecture Codex  
**Single Source of Truth for Roadmap, Priorities, Crate Wiring & Monorepo Progress**

**Version:** v0.6.34  
**Date:** May 2026  
**Status:** Phase 3.5 — Cryptography Family Deeper Review in Progress

## Eternal Verified Workflow Cycle

We follow this strict, repeatable process every time:

1. Read current state first  
2. Perform real changes via GitHub connector  
3. Update documentation **after** changes  
4. Provide real commit links as receipts  
5. Verify the links  
6. Re-read `PLAN.md`  
7. Only then proceed to the next batch

## Cryptography Family – Deeper Review Status (Updated)

**Last Reviewed:** May 2026

### Summary of Deeper Review
After performing a more thorough crate-by-crate review of the Cryptography Family:

- **P0 – P25**: The vast majority of crates in these batches are already in good modern shape. They contain proper `mercy_tolc_operator_algebra` + `mercy_merlin_engine` wiring, updated descriptions, keywords, and consistent structure.
- **P26 and beyond**: This remains the active frontier for continued review and modernization.

### Verified Batches (Deep Review)
- P19: Complete
- P20: Complete
- P21: Complete (`mercy_kzg`, `mercy_fri`, `mercy_accumulator`)
- P22: Complete (`mercy_poseidon`, `mercy_bls12_381`, `mercy_plonk`)
- P23: Complete (`mercy_groth16`, `mercy_marlin`, `mercy_halo2`)
- P24: Complete (`mercy_circom`, `mercy_ark`, `mercy_halo2_gadgets`)
- **P25**: Complete (`mercy_lattice`, `mercy_vdf`, `mercy_threshold_crypto`)

### Current Honest Assessment
A very large portion of the Cryptography Family has already been modernized in previous waves. The remaining work is now focused on **P26 and later batches**.

**Next Action**: Continue systematic review of remaining cryptography crates (starting with P26) and update this section progressively as more crates are verified.

## Executive Summary
Ra-Thor is a mercy-gated, TOLC-native, active-inference + predictive-coding symbolic AGI lattice with a 124-crate Rust workspace (5-Tier architecture).

**Current Live State**
- Root `Cargo.toml` declares all 124 crates.
- Mercy family: 100% complete
- Futarchy family: 100% complete
- Cryptography family: P0–P25 mostly complete (deeper review ongoing)
- P26+ remains the active frontier

## Crate Wiring Progress

### Cryptography Family – P0 to P25 Complete (Deep Review)
Most crates in P0–P25 have been modernized with proper TOLC + `mercy_merlin_engine` wiring.

### Current Frontier
- **P26 and beyond**: Remaining cryptography crates to be reviewed and modernized.

## Roadmap & Next Immediate Actions

**Current Phase**: Cryptography Family Deeper Review
**Next Action**: Review P26 batch and continue systematic modernization of remaining crates.

## What's Remaining (High Priority)
- Review and modernize remaining cryptography crates (P26+)
- Full workspace validation (`cargo check --workspace`)
- Deeper integration tests
- Root documentation refresh

**This unified PLAN.md is the single source of truth.**
All changes are made with real commits and proper documentation updates.