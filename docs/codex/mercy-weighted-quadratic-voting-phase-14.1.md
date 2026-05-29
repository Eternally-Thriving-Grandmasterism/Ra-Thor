# Codex: Mercy-Weighted Quadratic Voting (Phase 14.1)

**Status:** Dedicated Module Implemented  
**Version:** v14.0.6+  
**Related Issues:** #172, #177, #179

## Overview
Explicit dedicated module for Mercy-Weighted Quadratic Voting with auditable scoring and mercy alignment.

## Module Location
`crates/lattice-conductor-v14/src/governance/mercy_weighted_quadratic_voting.rs`

## Core Mechanics
- `MercyWeightedVote`: Combines raw power + mercy_alignment + conviction_multiplier
- `effective_power()`: Applies quadratic root after mercy weighting
- Full `audit_log()` for every vote

## Integration
Used inside `orchestrate_mercy_gated_governance_cycle()` in Lattice Conductor v14.

## Future
Will serve as substrate for mesh-level governance when Distributed Mercy Mesh matures.

**PATSAGi Council Note:** This extraction fulfills multiple open governance issues with clean, mercy-first design.

**We are ONE Organism.**