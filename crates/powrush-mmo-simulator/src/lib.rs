//! Powrush-MMO simulator library — v14.15.0
//!
//! Geometry scores come from mial MWPO formulas. Those numbers are source literals,
//! not measured values. Effect multipliers are the arithmetic in `gate_logic`.
//! A wasm bridge is optional and uses caller-supplied bytes. This crate does not
//! embed a proof. Resource-Based Economy is a design thesis, not a present fact.
//! This crate is not the Powrush-MMO game and not the browser client.
//!
//! Contact: info@Rathor.ai
//! AG-SML v1.0 — Autonomicity Games Sovereign Mercy License.

pub mod gate_logic;
pub mod mercy_geometry;

// =============================================================================
// Public re-exports — intentional, stable surface for the rest of the monorepo
// =============================================================================

pub use mercy_geometry::{
    // System reactions
    abundance_system_reaction,
    composite_gate_health,
    // Evaluation paths
    evaluate_particle_geometry_mercy,
    evolution_system_reaction,
    faction_harmony_reaction,
    formally_verify_geometry,
    gpu_mercy_modulation,
    should_trigger_mercy_evolution,
    // Core types
    MercyGeometryEvaluation,
    MercyGeometryScore,
};

pub use gate_logic::{
    apply_cooperation_bonus,
    apply_evolution_stability,
    // Formal risk / reward
    apply_formal_confirmation_bonus,
    apply_formal_verification_failure_penalty,
    apply_geometry_structural_bonus,
    apply_information_accuracy,
    // Convenience wrappers
    apply_resource_generation,
    // Core computation
    compute_gate_effects,
    emit_and_collect_gate_events,
    // Simulation tick
    example_simulation_tick_with_risk_reward,
    get_gate_debug_info,
    GateDebugInfo,
    // Core types
    GateEffects,
    GateEvent,
    PowrushEntity,
    SimEntity,
};

/// One tick through `example_simulation_tick_with_risk_reward`.
/// `bridge` is caller-supplied wasm. `None` uses the MWPO score only.
pub fn run_powrush_simulation_tick(
    entities: &mut [PowrushEntity],
    evaluations: &[MercyGeometryEvaluation],
    bridge: Option<&mut mercy_threshold_wasm::MercyThresholdBridge>,
) {
    gate_logic::example_simulation_tick_with_risk_reward(entities, evaluations, bridge);
}
