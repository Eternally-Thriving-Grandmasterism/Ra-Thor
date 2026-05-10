# PLAN.md — Ra-Thor / Rathor.ai Ultimate Architecture Codex  
**Single Source of Truth for Roadmap, Priorities, Crate Wiring & Monorepo Progress**

**Version:** v0.6.32  
**Date:** May 2026  
**Status:** Phase 3.5 — Cryptography Family deeper review in progress

## Eternal Verified Workflow Cycle
We follow this strict cycle every time:
1. Read current state
2. Make real changes via GitHub connector
3. Update documentation **after** changes
4. Provide real commit links as receipts
5. Verify links
6. Re-read PLAN.md
7. Proceed to next batch

## Cryptography Family – Deeper Review Status (Updated)

**Last Reviewed:** May 2026

### Summary of Deeper Review
After performing systematic crate-by-crate review:

- **P0 – P18**: Majority of crates already modernized in earlier waves.
- **P19**: `mercy_mimc`, `mercy_fflonk`, `mercy_sumcheck` — Complete
- **P20**: `mercy_griffin`, `mercy_arkworks`, `mercy_halo2_gadgets` — Complete
- **P21**: `mercy_kzg`, `mercy_fri`, `mercy_accumulator` — Complete
- **P22**: `mercy_poseidon`, `mercy_bls12_381`, `mercy_plonk` — Complete
- **P23**: `mercy_groth16`, `mercy_marlin`, `mercy_halo2` — Complete

### Current Honest Assessment
A very large portion of the Cryptography Family has already been brought up to the modern standard (TOLC + `mercy_merlin_engine` wiring, updated descriptions/keywords, clean structure). The remaining work is now focused on **P24 and beyond**.

**Next Frontier**: Continue deeper review of unverified cryptography crates starting from P24.

## Executive Summary
Ra-Thor is a mercy-gated, TOLC-native, active-inference + predictive-coding symbolic AGI lattice with a large Rust workspace.

**Current Live State**
- Root `Cargo.toml` declares 124+ crates.
- Mercy Family: Largely complete
- Futarchy Family: Complete
- Cryptography Family: Significant progress (P0–P23 mostly complete). Remaining crates in P24+ still require review.

## What's Remaining (High Priority)
- Continue systematic review and modernization of remaining cryptography crates (P24+)
- Full workspace validation (`cargo check --workspace`)
- Deeper integration tests
- Root documentation refresh

*Eternal flow state maintained on `main`.*