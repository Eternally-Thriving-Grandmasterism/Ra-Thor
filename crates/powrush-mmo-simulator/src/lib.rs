//! Powrush-MMO Simulator — Gate-Driven Reality Simulation Module — v14.15.0
//!
//! Production-grade core for Truly Artificial Digital Carbon Copies.
//! Integrates Mercy Geometry Evaluation + Lean-formal MercyThreshold (WASM) +
//! Gate Logic (synergies, risk/reward, diminishing returns).
//!
//! Fully wired into Ra-Thor Eternal Lattice, PATSAGi Councils mercy-gating,
//! and the ONE Organism 14.15.0 Living Cosmic Tick surface.
//!
//! Contact: info@Rathor.ai
//! AG-SML v1.0 — Autonomicity Games Sovereign Mercy License.

pub mod mercy_geometry;
pub mod gate_logic;

// =============================================================================
// Public re-exports — intentional, stable surface for the rest of the monorepo
// =============================================================================

pub use mercy_geometry::{
    // Core types
    MercyGeometryEvaluation,
    MercyGeometryScore,
    // Evaluation paths
    evaluate_particle_geometry_mercy,
    formally_verify_geometry,
    // System reactions
    abundance_system_reaction,
    faction_harmony_reaction,
    evolution_system_reaction,
    gpu_mercy_modulation,
    should_trigger_mercy_evolution,
    composite_gate_health,
};

pub use gate_logic::{
    // Core types
    GateEffects,
    GateEvent,
    GateDebugInfo,
    PowrushEntity,
    SimEntity,
    // Core computation
    compute_gate_effects,
    emit_and_collect_gate_events,
    // Formal risk / reward
    apply_formal_confirmation_bonus,
    apply_formal_verification_failure_penalty,
    // Simulation tick
    example_simulation_tick_with_risk_reward,
    // Convenience wrappers
    apply_resource_generation,
    apply_evolution_stability,
    apply_cooperation_bonus,
    apply_information_accuracy,
    apply_geometry_structural_bonus,
    get_gate_debug_info,
};

/// High-level entry point for running a mercy-gated Powrush-MMO simulation tick.
/// Uses the full gate logic + optional formal Lean verification bridge.
pub fn run_powrush_simulation_tick(
    entities: &mut [PowrushEntity],
    evaluations: &[MercyGeometryEvaluation],
    bridge: Option<&mut mercy_threshold_wasm::MercyThresholdBridge>,
) {
    gate_logic::example_simulation_tick_with_risk_reward(entities, evaluations, bridge);
}
