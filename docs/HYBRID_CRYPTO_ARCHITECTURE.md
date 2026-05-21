# Hybrid Classical + Post-Quantum Cryptography Architecture

## End-to-End Example Flow: Private Threshold Council Decision

**Scenario**: A group of councils wants to make a sensitive decision with the following properties:
- Structured deliberation
- Efficient signature aggregation
- Post-quantum secure channel
- Private proof of threshold approval

**Flow**:

1. **Deliberation** (`patsagi_deliberation.rs`)
   - Councils exchange Endorsements and Concerns.
   - A preliminary consensus is reached.

2. **BLS Aggregation** (`bls_aggregation.rs`)
   - High-weight councils produce BLS signatures on the decision.
   - Signatures are aggregated for efficiency.

3. **ML-KEM Secure Channel** (`ml_kem.rs`)
   - ML-KEM is used to establish a post-quantum secure channel for sharing sensitive partial data.

4. **STARK Proof** (`starks.rs`)
   - A STARK proof is generated to prove that a sufficient threshold of councils approved the decision without revealing individual positions.

5. **Future: Lattice Threshold Signatures** (`lattice_threshold_signatures.rs`)
   - Long-term migration path to fully post-quantum threshold signing.

This flow demonstrates how the modules cross-pollinate to support advanced, privacy-preserving, and future-proof governance.