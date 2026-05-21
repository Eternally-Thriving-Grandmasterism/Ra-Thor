# Hybrid Classical + Post-Quantum Cryptography Architecture

## Cross-Module Interaction Example

**Scenario: Private Threshold Council Decision**

1. Councils deliberate using `patsagi_deliberation.rs`.
2. They produce partial BLS signatures (`bls_aggregation.rs`).
3. When higher security is needed, ML-KEM (`ml_kem.rs`) is used to establish a secure channel for sharing partial signatures.
4. A STARK proof (`starks.rs`) can be generated to prove that a threshold was met without revealing individual votes.
5. In the long term, lattice threshold signatures can replace BLS for full post-quantum security.

This shows how the modules cross-pollinate to support advanced governance use cases.