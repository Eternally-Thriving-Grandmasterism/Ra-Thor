//! Mercy Geometry Integration for Lattice Conductor
//!
//! Wires the hybrid Mercy Threshold (Lean WASM + native mial) into
//! geometry evolution and self-evolution decisions.

use mial::mwpo::{GeometryParams, MercyContext, MercyGeometryScore};

/// High-level mercy geometry check used by Lattice Conductor
/// before applying geometry updates, epigenetic blessings, or evolution steps.
///
/// Uses the hybrid path: prefers native mial implementation when available,
/// falls back to formal Lean WASM threshold when needed.
pub fn check_mercy_geometry_before_evolution(
    geometry: &GeometryParams,
    context: &MercyContext,
    council_id: u32,
) -> Result<MercyGeometryScore, anyhow::Error> {
    // When mial feature is enabled, this calls the full production implementation
    // (which itself can fall back to or combine with the WASM bridge).
    mial::mwpo::MercyWeightedPreferenceOptimization::new()
        .evaluate_geometry_mercy_component(geometry, context, council_id)
        .map_err(|e| anyhow::anyhow!("Mercy geometry check failed: {}", e))
}

/// Convenience wrapper for Powrush-MMO style calls.
pub fn is_geometry_mercy_safe(
    geometry: &GeometryParams,
    context: &MercyContext,
    council_id: u32,
) -> bool {
    check_mercy_geometry_before_evolution(geometry, context, council_id)
        .map(|score| score.overall >= 0.92)
        .unwrap_or(false)
}