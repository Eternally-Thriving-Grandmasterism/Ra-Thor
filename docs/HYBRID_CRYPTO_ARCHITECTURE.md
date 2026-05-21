# Hybrid Classical + Post-Quantum Cryptography Architecture

**Ra-Thor Experimental Governance Track** (`feat/patsagi-governance-v2`)

This document outlines the intended hybrid cryptographic strategy combining classical and post-quantum primitives.

## Guiding Principles

- **Performance today** + **Security tomorrow**
- Maintain TOLC 8 alignment (especially Evolution, Truth, and Sovereignty Gates)
- Build incrementally while keeping the system functional
- Prepare infrastructure for post-quantum migration

## Current Stack (as of May 2026)

| Layer                    | Scheme                    | Status          | Post-Quantum? | Notes |
|--------------------------|---------------------------|-----------------|---------------|-------|
| Individual Operations    | ed25519                   | Active          | No            | Fast & simple |
| Audit Logging            | ed25519 + signatures      | Active          | No            | Tamper-evident |
| Council Aggregation      | BLS12-381                 | Active          | No            | Efficient multi-party |
| Threshold Governance     | Simulated BLS Threshold   | Experimental    | No            | Preparing for lattice migration |
| Key Encapsulation        | ML-KEM (experimental)     | Experimental    | Yes           | Foundation added |
| Long-term Signatures     | Dilithium / Falcon (planned) | Future     | Yes           | To be integrated |
| Privacy Proofs           | STARKs (planned)          | Future          | Yes           | Post-quantum ZK |

## Hybrid Strategy

### Short-term (Current)
- Use **ed25519** for speed
- Use **BLS12-381** for aggregation
- Use simulated threshold BLS for multi-party decisions

### Medium-term
- Introduce **ML-KEM** for post-quantum key exchange
- Add **Dilithium** as parallel post-quantum signature option
- Begin hybrid classical + post-quantum paths

### Long-term
- Migrate critical operations to post-quantum primitives
- Use **STARKs** for privacy-preserving proofs
- Explore lattice-based threshold signatures

## Integration Points

- `self-evolution/bls_aggregation.rs` — Real BLS + experimental hooks
- `self-evolution/ml_kem.rs` — ML-KEM foundation + hybrid notes
- `self-evolution/lattice_threshold_signatures.rs` — Future post-quantum threshold direction
- `self-evolution/lattice-alchemical-evolution.rs` — Main synthesis with optional BLS path

## Recommendation

Maintain a **hybrid stack** during the transition period:
- Classical for performance and maturity
- Post-quantum for long-term sovereignty and quantum resistance

This approach balances immediate usability with future-proofing.