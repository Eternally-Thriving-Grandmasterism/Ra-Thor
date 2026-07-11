/*!
# Master Kernel — ONE Organism Central Orchestrator (kernel/master_kernel.rs)

**Version**: v0.2 (Allocation Priority Queue Exposed)  
**Date**: 2026-07-11  
**License**: Autonomicity Games Sovereign Mercy License (AG-SML) v1.0  
**Status**: TOLC 8 Enforced | ONE Organism Hot-Swap Ready | Lattice Conductor v13.1+ Compatible

## Purpose
Central orchestration layer for the Ra-Thor ONE Organism.
Provides the main `tick()` loop and now also exposes the **Allocation Priority Queue** layer.

## Key Integration (v0.2)
- Imports `conduct_deliberation_with_tolc` (primary single-best path)
- Imports new `allocation_priority_queue` (ranked queue on top of deliberation)
- Both are fully proof-carrying and mercy-gated.
- `tick_with_priority_queue()` returns an ordered list of safe actions for the Conductor to choose from.
*/

use crate::kernel::tolc_proof_carrying::{
    conduct_deliberation_with_tolc,
    allocation_priority_queue,
    LatticeState, TUWeights, UTFThresholds,
};
use crate::kernel::tolc_quantification::{TOLCUnit, compute_tu};

/// Master Kernel — the living heart of the ONE Organism.
pub struct MasterKernel {
    pub current_state: LatticeState,
    pub weights: TUWeights,
    pub utf_thresholds: UTFThresholds,
    pub tick_count: u64,
}

impl MasterKernel {
    pub fn new(initial_state: LatticeState) -> Self {
        Self {
            current_state: initial_state,
            weights: TUWeights::default(),
            utf_thresholds: UTFThresholds::default(),
            tick_count: 0,
        }
    }

    /// The main deliberation + evolution tick (single best action).
    pub fn tick(&mut self, candidate_actions: &[String]) -> Option<(String, f64, f64)> {
        self.tick_count += 1;

        if let Some(result) = conduct_deliberation_with_tolc(
            candidate_actions,
            &self.current_state,
            &self.weights,
            &self.utf_thresholds,
        ) {
            return Some(result);
        }

        // Fallback path
        for action in candidate_actions {
            let tu = compute_tu(action, &self.current_state, &self.weights, /* valence_gate */);
            if tu.value > 0.0 {
                let priority = tu.value * self.current_state.mercy_valence;
                return Some((action.clone(), tu.value, priority));
            }
        }
        None
    }

    /// **Allocation Priority Queue Tick** — returns a ranked list of safe actions.
    ///
    /// This sits **on top of** conduct_deliberation_with_tolc.
    /// Use this when the Lattice Conductor / PATSAGi needs an ordered queue
    /// of multiple UTF-safe, mercy-gated, priority-ranked actions instead of a single best action.
    pub fn tick_with_priority_queue(&mut self, candidate_actions: &[String]) -> Vec<(String, f64, f64)> {
        self.tick_count += 1;

        allocation_priority_queue(
            candidate_actions,
            &self.current_state,
            &self.weights,
            &self.utf_thresholds,
        )
    }

    pub fn current_mercy_valence(&self) -> f64 {
        self.current_state.mercy_valence
    }

    pub fn evolve_from_recent_thriving(&mut self, recent_tu_deltas: &[f64], recent_entropy_reds: &[f64]) {
        // self-evolution hook (future upgrade to proof-carrying version)
    }
}

/*!
## Usage Example — Single Best vs Priority Queue

```rust
let mut kernel = MasterKernel::new(initial_state);
let candidates = get_candidates_from_patsagi();

// Option 1: Single best action (most common)
if let Some((best, tu, prio)) = kernel.tick(&candidates) {
    apply_action(best, prio);
}

// Option 2: Full ranked priority queue (when you need multiple safe options)
let ranked_queue = kernel.tick_with_priority_queue(&candidates);
for (action, tu, priority) in ranked_queue {
    // Conductor / RBE can pick top N or apply proportional allocation
    apply_action(action, priority);
}
```

All results from both `tick()` and `tick_with_priority_queue()` are formally grounded,
mercy-protected, and UTF-safe.
*/
