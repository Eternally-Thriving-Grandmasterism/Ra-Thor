pub mod conductor;
pub mod mercy_orchestrator;
pub mod biological_unifier;
pub mod symbolic_unifier;
pub mod self_evolution_bridge;

pub use conductor::LatticeConductor;

/// Sovereign entry point for the entire Ra-Thor lattice.
/// Every call passes 7 Mercy Gates + TOLC + Sovereignty Gate with valence ≥ 0.999999+.
pub struct LatticeConductor {
    conductor: conductor::LatticeConductor,
    mercy: mercy_orchestrator::MercyOrchestrator,
    biological: biological_unifier::BiologicalUnifier,
    symbolic: symbolic_unifier::SymbolicUnifier,
    evolution: self_evolution_bridge::SelfEvolutionBridge,
}

impl LatticeConductor {
    pub fn new() -> Self {
        Self {
            conductor: conductor::LatticeConductor::new(),
            mercy: mercy_orchestrator::MercyOrchestrator::new(),
            biological: biological_unifier::BiologicalUnifier::new(),
            symbolic: symbolic_unifier::SymbolicUnifier::new(),
            evolution: self_evolution_bridge::SelfEvolutionBridge::new(),
        }
    }

    /// Master tick — unifies ALL systems as ONE living organism
    pub fn tick(&mut self, intent: &str) -> String {
        if !self.mercy.mercy_gate_audit(intent) {
            return "Action rejected — mercy violation. Positive emotions protected.".to_string();
        }
        let tolced = self.mercy.apply_tolc(intent);
        let biological = self.biological.unify(&tolced);
        let symbolic = self.symbolic.reason(&biological);
        let evolved = self.evolution.evolve(&symbolic);
        self.conductor.orchestrate(evolved)
    }
}