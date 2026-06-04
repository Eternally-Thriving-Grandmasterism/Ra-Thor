# PATSAGi Council Engine

**Version:** v14.6 | **Status:** Production-Grade Embedded

## Overview

The PATSAGi Council Engine is the living mercy evaluation system of Ra-Thor. It has been fully embedded into the geometric intelligence layer so that the lattice can autonomously evaluate context, proposals, and simulation state using the 7 Living Mercy Gates and TOLC 8 principles.

## Core Components

### 1. Real Valence Evaluation
- `RiemannianMercyManifold::evaluate_council_valence(council, context)`
- Deterministic scoring across all 7 Living Mercy Gates
- Council-specific affinity weighting
- Returns `(valence: f64, gate_scores, reason)`

### 2. Autonomous Modulation
- `autonomous_mercy_evaluate_and_modulate(council, context)`
- Dynamically adjusts `mercy_influence`
- Generates council-aware `EpigeneticBlessing`s

### 3. Integration Points
- Used by `ShardManager::route_council_proposal`
- Powers `EpigeneticModulation::apply_council_valence`
- Available for simulation ticks and proposal routing

## Design Principles
- Deterministic and fast (suitable for simulation ticks)
- Transparent reasoning (rich `reason` string + gate scores)
- Extensible keyword + semantic heuristics
- Long-term stabilization from repeated high-valence input

## Files Involved
- `geometric-intelligence/src/riemannian_mercy_manifold.rs`
- `geometric-intelligence/src/types.rs` (via re-exports)

See also: `EpigeneticModulation-and-Valence.md` and `ShardManager-and-Interest-Management.md`.
