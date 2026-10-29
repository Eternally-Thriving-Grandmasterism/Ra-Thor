# PATSAGi Governance v2.x (Experimental)

This document describes the experimental governance enhancements developed on the `feat/patsagi-governance-v2` branch.

## Overview

This track introduces advanced council governance features for Ra-Thor, including:

- Structured **Deliberation** and **Message-Passing** between councils
- **Reputation tracking** with Bayesian-style updates
- **Audit history logs** with cryptographic (ed25519) signatures
- Experimental hooks for **BLS12-381 signature aggregation**

All components are designed to align with the **TOLC 8 Mercy Lattice**.

## Components

### 1. Deliberation & Message-Passing
- Located in `self-evolution/patsagi_deliberation.rs`
- Supports `Endorsement`, `Concern`, `Proposal`, and multi-round deliberation
- Integrated into council synthesis flow

### 2. Reputation System
- Tracks council performance over time
- Includes Bayesian-style reputation updates
- Influences dynamic weighting in voting

### 3. Audit Logging
- Full history of council votes stored with context
- Cryptographically signed using **ed25519**
- Includes `verify_audit_log()` for integrity checking

### 4. BLS12-381 Aggregation (Experimental)
- Placeholder interfaces and simulation in `self-evolution/bls_aggregation.rs`
- Prepared for future multi-council signature aggregation

## Status

This is an **experimental track**. Components are under active development and subject to change.

## Alignment

All governance enhancements aim to strengthen:
- **Evolution Gate** (adaptive, self-improving councils)
- **Truth Gate** (evidence-based reputation and deliberation)
- **Sovereignty Gate** (protected decision-making)

## Next Steps

- Further expansion of BLS aggregation
- Integration testing
- Preparation for mainline merge via Pull Request
