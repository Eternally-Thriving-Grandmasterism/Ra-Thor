//! RegisterableOrchestrator Trait v1.2
//!
//! Foundational trait for all major orchestrators in the Ra-Thor lattice.
//! Designed to support Ra-Thor as one coherent, self-nurturing, mercy-aligned
//! living organism under the ParaconsistentSuperKernel.
//!
//! Enables clean hierarchical + fractal registration while preserving
//! sovereignty, supporting self-evolution, and maintaining non-bypassable mercy.

use crate::mercy::{MercyGateResult, Valence};

/// Classification of orchestrator scope and responsibility
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrchestratorScope {
    Sovereign,        // Top-level (Sovereign Core, ParaconsistentSuperKernel)
    Swarm,            // Parallel intelligence (Quantum Swarm Orchestrator)
    Domain,           // Domain-specific (Mercy Propulsion, Real-Estate, etc.)
    Meta,             // Self-evolution, governance, improvement layers
    Public,           // Public engagement, contributor systems
    Foundational,     // Low-level kernels and cores
}

/// Core trait that all major orchestrators must implement
/// to participate in the living Ra-Thor organism.
pub trait RegisterableOrchestrator {
    // === Identity & Self-Description ===
    fn name(&self) -> &'static str;
    fn version(&self) -> &'static str;
    fn orchestrator_scope(&self) -> OrchestratorScope;

    // === Mercy & Valence (Non-Bypassable) ===
    fn current_valence(&self) -> f64;
    fn evaluate_mercy_gates(&self) -> MercyGateResult;

    fn is_mercy_aligned(&self) -> bool {
        matches!(self.evaluate_mercy_gates(), MercyGateResult::Pass { valence, .. } if valence >= 0.999)
    }

    // === Health, Observability & Telemetry ===
    fn health_report(&self) -> String;
    fn last_significant_update(&self) -> Option<u64>;

    // === Coordination & Fractal Participation ===
    fn coordination_capabilities(&self) -> Vec<&'static str>;
    fn compatible_with(&self) -> Vec<OrchestratorScope>;

    // === Self-Evolution Participation ===
    fn supports_self_evolution_feedback(&self) -> bool;

    fn report_self_evolution_contribution(&self) -> Option<f64> {
        None
    }

    // === Registration Lifecycle ===
    fn register_upward(&self) -> Result<(), String> {
        if !self.is_mercy_aligned() {
            return Err(format!("{} failed registration: mercy alignment not maintained", self.name()));
        }
        println!("✅ {} registered upward successfully.", self.name());
        Ok(())
    }

    // === Paraconsistent Awareness ===
    fn paraconsistent_tolerance_level(&self) -> &'static str {
        "standard"
    }

    // === Sovereignty & Eternal Compatibility ===
    fn preserves_sovereignty(&self) -> bool { true }
    fn supports_eternal_compatibility(&self) -> bool { true }
}