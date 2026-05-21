# Hybrid Classical + Post-Quantum Cryptography Architecture

**Ra-Thor Experimental Governance Track** (`feat/patsagi-governance-v2`)

## Module Interaction Overview

This section describes how the different cryptographic and governance modules can interact:

- **BLS Aggregation** (`bls_aggregation.rs`): Used for efficient multi-council consensus and signature aggregation.
- **ML-KEM** (`ml_kem.rs`): Provides post-quantum key encapsulation for secure channels between councils.
- **STARKs** (`starks.rs`): Future post-quantum zero-knowledge layer for private but verifiable proofs (e.g. private participation or threshold approval).
- **Lattice Threshold Signatures** (`lattice_threshold_signatures.rs`): Long-term replacement path for post-quantum t-of-n decision making.
- **Deliberation** (`patsagi_deliberation.rs`): Provides structured messaging and consensus building that can feed into cryptographic layers.
- **Reputation** (`lattice-alchemical-evolution.rs`): Can influence weighting and trust in cryptographic operations.

## Example Interaction Flow

1. Councils deliberate using the deliberation module.
2. High-weight or trusted councils generate BLS signatures.
3. When higher security is needed, ML-KEM can be used to establish secure channels.
4. For private proofs, STARKs can be used to prove participation or threshold approval without revealing identities.
5. In the long term, lattice-based threshold signatures can replace or complement BLS threshold mechanisms.

See the main sections below for the full strategy.