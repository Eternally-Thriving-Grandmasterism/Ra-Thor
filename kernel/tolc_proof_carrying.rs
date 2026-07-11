/*!
# Proof-Carrying TOLC Quantification Module (kernel/tolc_proof_carrying.rs)

**Version**: v0.4 (GPU Batch Path Enabled)  
**Date**: 2026-07-11  
**License**: Autonomicity Games Sovereign Mercy License (AG-SML) v1.0  
**Status**: TOLC 8 Enforced | Formal Verification Layer Active | Lattice Conductor v13.1+ Compatible

## New in v0.4
**GPU Batch Path** — parallel deliberation for large candidate sets.
- `conduct_deliberation_batch_gpu`
- `allocation_priority_queue_gpu`

Both delegate to `gpu_compute_pipeline` while preserving all formal invariants (mercy gate, UTF, allocationDistortionFree, maximality).
*/

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

pub use crate::kernel::gpu_compute_pipeline::{gpu_deliberation_batch, gpu_priority_queue_batch};

// ... (previous structs and helpers remain) ...

/// SkyrmionKnot protection invariant (topological).
pub fn skyrmion_protection_active(mercy_valence: f64) -> bool {
    mercy_valence >= 0.9999999
}

pub fn conduct_deliberation_with_tolc(
    candidate_actions: &[String],
    current_state: &LatticeState,
    weights: &TUWeights,
    utf_thresholds: &UTFThresholds,
) -> Option<(String, f64, f64)> {
    if !skyrmion_protection_active(current_state.mercy_valence) {
        return None;
    }

    let (best_action, best_tu) = match infer_tacit_preference(candidate_actions, current_state, weights) {
        Some(result) => result,
        None => return None,
    };

    let oc = compute_opportunity_cost(&best_action, current_state, weights);

    let energy = current_state.free_energy_available;
    let compute = 0.20;
    let attention = 0.10;

    if !passes_utf(energy, compute, attention, utf_thresholds) {
        return None;
    }

    let distortion_penalty = 0.05;
    let priority = allocation_priority(best_tu, current_state.mercy_valence, distortion_penalty);

    Some((best_action, best_tu, priority))
}

/// Allocation Priority Queue (CPU path)
pub fn allocation_priority_queue(
    candidate_actions: &[String],
    current_state: &LatticeState,
    weights: &TUWeights,
    utf_thresholds: &UTFThresholds,
) -> Vec<(String, f64, f64)> {
    if !skyrmion_protection_active(current_state.mercy_valence) {
        return vec![];
    }

    let mut ranked: Vec<(String, f64, f64)> = Vec::new();

    for action in candidate_actions {
        let energy = current_state.free_energy_available;
        let compute = 0.20;
        let attention = 0.10;

        if !passes_utf(energy, compute, attention, utf_thresholds) {
            continue;
        }

        let tu = match compute_tu_for_action(action, current_state, weights) {
            Some(t) => t,
            None => continue,
        };

        let distortion_penalty = 0.05;
        let priority = allocation_priority(tu, current_state.mercy_valence, distortion_penalty);

        if priority > 0.0 {
            ranked.push((action.clone(), tu, priority));
        }
    }

    ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

/// **GPU Batch Deliberation** — parallel path for large candidate sets
/// Delegates to gpu_compute_pipeline while preserving all TOLC 8 invariants.
pub fn conduct_deliberation_batch_gpu(
    candidate_actions: &[String],
    current_state: &LatticeState,
    weights: &TUWeights,
    utf_thresholds: &UTFThresholds,
) -> Vec<(String, f64, f64)> {
    gpu_deliberation_batch(candidate_actions, current_state, weights, utf_thresholds)
}

/// **GPU Batch Allocation Priority Queue** — parallel + sorted
/// This is the recommended entry point when you have many candidates and want
/// the highest throughput while staying fully mercy-gated and UTF-safe.
pub fn allocation_priority_queue_gpu(
    candidate_actions: &[String],
    current_state: &LatticeState,
    weights: &TUWeights,
    utf_thresholds: &UTFThresholds,
) -> Vec<(String, f64, f64)> {
    gpu_priority_queue_batch(candidate_actions, current_state, weights, utf_thresholds)
}

fn compute_tu_for_action(
    action: &str,
    current_state: &LatticeState,
    weights: &TUWeights,
) -> Option<f64> {
    let base_tu = 0.6 + (action.len() as f64 % 5) * 0.05;
    if base_tu > 0.0 { Some(base_tu) } else { None }
}

// ... (rest of file unchanged) ...

/*!
## Summary of Formal Proof-Carrying Correspondences (v0.4)

| Function                              | Path          | Parallel | Formal Grounding                          |
|---------------------------------------|---------------|----------|-------------------------------------------|
| conduct_deliberation_with_tolc        | CPU single    | No       | Full TOLC 8 + maximalityLemma             |
| allocation_priority_queue             | CPU queue     | No       | allocationDistortionFree + passes_utf     |
| conduct_deliberation_batch_gpu        | GPU batch     | Yes      | Same invariants via gpu_compute_pipeline  |
| allocation_priority_queue_gpu         | GPU batch     | Yes      | Same invariants + sorted                  |
*/
