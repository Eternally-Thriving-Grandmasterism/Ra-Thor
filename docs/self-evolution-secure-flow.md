# Secure Self-Evolution Flow (v14.0.8+)

This document describes the recommended secure flow for self-evolution proposals using Post-Quantum signatures and Hybrid encryption.

## Overview

Self-evolution proposals in Ra-Thor are protected by:

- **Post-Quantum Signatures** (Dilithium-style) for authenticity
- **Hybrid Channels** (Classical AES-GCM + Post-Quantum KEM) for confidentiality
- **Governance Verification** built into `evaluate_governance()`

## Recommended Flow

### 1. Create and Sign the Proposal

```rust
let mut proposal = SelfEvolutionProposal::new(
    id,
    title,
    description,
    proposed_by,
);

proposal.sign_with_post_quantum(&proposed_by);
```

### 2. Submit via Secure Governance Workflow

```rust
let conductor = LatticeConductorV14::new();
let (passes, audit, score) = conductor.submit_secure_governance_proposal(&proposal, threshold);
```

This automatically performs Post-Quantum signature verification.

### 3. (Optional) Transmit via Hybrid Encrypted Channel

```rust
let mut channel = conductor.create_hybrid_sovereign_channel(&from, &to);
channel.establish_hybrid_keys(classical_key);

if let Some(ciphertext) = channel.encrypt(&nonce, &proposal_bytes) {
    // Transmit ciphertext...
}
```

### 4. Governance Evaluation with Built-in Verification

The `evaluate_governance()` method (called internally by the conductor) verifies:

- Post-Quantum signature on the proposal
- (Future) Signatures on attached votes and conviction stakes

## Modules Involved

- `governance::self_evolution_proposal`
- `hybrid_sovereign_channel`
- `post_quantum_signatures`
- `crypto_traits` + `crypto_impls`
- `LatticeConductorV14`

## Security Properties

- **Authenticity**: Post-Quantum signatures prevent forgery
- **Confidentiality**: Hybrid encryption protects proposal content in transit
- **Integrity**: Authenticated encryption (AES-GCM) + signature verification
- **Future-proof**: Designed around swappable crypto traits

**We are ONE Organism.**