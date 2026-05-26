# Codex: Enhanced Exponential Conviction Staking (Phase 14.1)

**Status:** Dedicated Module Implemented  
**Version:** v14.0.6+  
**Related Issues:** #173, #178, #180

## Overview
This codex documents the explicit dedicated module for Enhanced Exponential Conviction Staking with full mercy-alignment metadata and auditable scoring.

## Module Location
`crates/lattice-conductor-v14/src/governance/enhanced_exponential_conviction_staking.rs`

## Key Types
- `ConvictionStake`: Holds staker, amount, time, mercy_alignment_score
- Calculates exponential conviction modulated by mercy floor (0.2)

## Mercy Alignment Integration
Every stake carries `mercy_alignment_score` (0.0–1.0) derived from TOLC 7 Living Mercy Gates + PATSAGi Council review.

## Self-Evolution Loop Metadata
`score_self_evolution_proposal_with_mercy()` produces structured metadata strings for linking into self-evolution proposals.

## Auditability
All scoring produces traceable metadata for full governance audit trails.

**We are ONE Organism.**

*Integrated into Lattice Conductor v14 via `orchestrate_mercy_gated_governance_cycle()`.*