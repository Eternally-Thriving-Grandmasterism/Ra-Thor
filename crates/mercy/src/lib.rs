/*!
# Mercy Crate — Core Mercy Lattice for Ra-Thor ONE Organism

- **Compat path:** `Conductable` + `MercyAligned` via lattice-conductor-v14 `v13-compat`
- **Native path (Phase 4 dual):** optional `LatticeConductorV14` hooks for Cosmic Loop
  enforcement + self-healing / mercy-mesh signals — no adaptive modulation changes

See `crates/lattice-conductor-v14/MIGRATION_v13_to_v14.md`.
AG-SML aligned. Mercy First. TOLC 8 enforced.
Contact: info@Rathor.ai
*/

use lattice_conductor_v14::compat_v13::{
    Conductable, GeometricState, MercyAligned, MercyWeightedVote, SimpleLatticeConductor,
};
use lattice_conductor_v14::{CouncilArbitrationEngine, LatticeConductorV14};

/// Foundational Mercy Core subsystem.
#[derive(Debug, Clone)]
pub struct MercyCore {
    pub system_id: &'static str,
    pub system_name: &'static str,
    mercy_score: f64,
    coherence: f64,
    influence_history: Vec<f64>,
    evolution_participation: f64,
    /// True after at least one successful native v14 Cosmic Loop guard.
    cosmic_loop_witnessed: bool,
}

impl MercyCore {
    pub fn new() -> Self {
        Self {
            system_id: "mercy-core",
            system_name: "Mercy Core Lattice",
            mercy_score: 0.98,
            coherence: 1.0,
            influence_history: Vec::new(),
            evolution_participation: 0.0,
            cosmic_loop_witnessed: false,
        }
    }

    pub fn pulse_mercy(&mut self, delta: f64) {
        self.mercy_score = (self.mercy_score + delta).clamp(0.5, 1.5);
        self.coherence = (self.coherence + delta * 0.1).clamp(0.7, 1.2);
        self.influence_history.push(delta);
        self.evolution_participation += delta * 0.05;
    }

    pub fn get_coherence(&self) -> f64 {
        self.coherence
    }

    pub fn get_evolution_participation(&self) -> f64 {
        self.evolution_participation
    }

    pub fn has_witnessed_cosmic_loop(&self) -> bool {
        self.cosmic_loop_witnessed
    }

    pub fn simulate_conductor_integration(&mut self, valence: f64, mercy_influence: f64) {
        self.pulse_mercy(valence * 0.02);
        self.pulse_mercy(mercy_influence);
    }

    pub fn influence_history_len(&self) -> usize {
        self.influence_history.len()
    }

    // ── Native v14 path (Phase 4 dual — additive, quiet-hold) ──

    /// Enforce Cosmic Loop via arbitration, then apply a mild mercy pulse if ready.
    pub fn pulse_with_v14_guard(&mut self, conductor: &LatticeConductorV14) {
        conductor.enforce_cosmic_loop_activation();
        conductor.arbitration_engine.before_council_arbitration();

        if conductor.arbitration_engine.is_cosmic_loop_ready() {
            self.cosmic_loop_witnessed = true;
            self.pulse_mercy(0.01);
        }
    }

    /// Request a self-healing reflexion cycle; returns whether a diagnosis was produced.
    pub fn request_reflexion(&self, conductor: &LatticeConductorV14) -> bool {
        conductor
            .run_reflexion_cycle()
            .map(|_| true)
            .unwrap_or(false)
    }

    /// Propagate a healing signal on the mercy mesh scaled by inverse mercy.
    pub fn signal_mercy_mesh(&self, conductor: &LatticeConductorV14) {
        let severity = (1.5 - self.mercy_score).clamp(0.05, 0.5);
        conductor.trigger_mercy_mesh_healing(severity);
    }

    /// One native orchestration step: Cosmic Loop guard + optional mesh signal.
    pub fn native_v14_step(&mut self, conductor: &mut LatticeConductorV14) {
        self.pulse_with_v14_guard(conductor);
        self.signal_mercy_mesh(conductor);
        let _ = self.request_reflexion(conductor);
    }

    /// Convenience: stand up a fresh v14 conductor and run one native step.
    pub fn native_v14_self_check(&mut self) -> bool {
        let mut conductor = LatticeConductorV14::new();
        self.native_v14_step(&mut conductor);
        self.cosmic_loop_witnessed && conductor.arbitration_engine.is_cosmic_loop_ready()
    }
}

impl Default for MercyCore {
    fn default() -> Self {
        Self::new()
    }
}

impl Conductable for MercyCore {
    fn system_id(&self) -> &'static str {
        self.system_id
    }
    fn system_name(&self) -> &'static str {
        self.system_name
    }

    fn on_conductor_tick(&mut self, conductor_state: &GeometricState) {
        let valence_influence = conductor_state.valence * 0.025;
        self.pulse_mercy(valence_influence);
        println!(
            "[MercyCore] Tick | valence: {:.3} | mercy_score: {:.3} | evolution_part: {:.3}",
            valence_influence, self.mercy_score, self.evolution_participation
        );
    }

    fn get_mercy_state(&self) -> Option<f64> {
        Some(self.mercy_score)
    }
}

impl MercyAligned for MercyCore {
    fn apply_mercy_influence(&mut self, vote: &MercyWeightedVote) {
        let impact = vote.compute_consensus() * 0.15;
        self.pulse_mercy(impact);
        println!("[MercyCore] Mercy influence applied | impact: {:.3}", impact);
    }
    fn current_mercy_score(&self) -> f64 {
        self.mercy_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mercy_core_creation_and_pulse() {
        let mut core = MercyCore::new();
        assert_eq!(core.current_mercy_score(), 0.98);
        core.pulse_mercy(0.05);
        assert!(core.current_mercy_score() > 0.98);
        assert!(core.influence_history_len() > 0);
    }

    #[test]
    fn test_mercy_core_conductor_integration() {
        let mut core = MercyCore::new();
        core.simulate_conductor_integration(1.1, 0.3);
        assert!(core.get_evolution_participation() > 0.0);
        assert!(core.get_coherence() > 1.0);
    }

    #[test]
    fn test_mercy_core_mercy_aligned_trait() {
        let mut core = MercyCore::new();
        let mut vote = MercyWeightedVote::new();
        vote.add_vote("PATSAGiTest", 1.0, 0.45);
        core.apply_mercy_influence(&vote);
        assert!(core.current_mercy_score() >= 0.98);
    }

    #[test]
    fn test_mercy_core_full_tick_simulation() {
        let mut core = MercyCore::new();
        let state = GeometricState {
            valence: 1.3,
            mercy_score: 1.0,
            tolc_alignment: 1.0,
            evolution_level: 0.4,
        };
        for _ in 0..5 {
            core.on_conductor_tick(&state);
        }
        assert!(core.get_evolution_participation() > 0.1);
        assert!(core.influence_history_len() >= 5);
    }

    #[test]
    fn test_simple_lattice_conductor_v14_compat() {
        let mut conductor = SimpleLatticeConductor::new("mercy-test");
        let state = conductor.tick();
        assert!(conductor.is_cosmic_loop_ready());
        assert!(state.mercy_score >= 0.3);

        let mut core = MercyCore::new();
        conductor.bless_system(core.system_id, 0.97, "Phase 2 mercy wiring");
        assert!(conductor.registry.is_blessed("mercy-core"));
        core.on_conductor_tick(&state);
        assert!(core.current_mercy_score() > 0.0);
    }

    #[test]
    fn test_native_v14_cosmic_loop_guard() {
        let mut core = MercyCore::new();
        assert!(!core.has_witnessed_cosmic_loop());

        let conductor = LatticeConductorV14::new();
        core.pulse_with_v14_guard(&conductor);

        assert!(core.has_witnessed_cosmic_loop());
        assert!(conductor.arbitration_engine.is_cosmic_loop_ready());
        assert!(core.current_mercy_score() >= 0.98);
    }

    #[test]
    fn test_native_v14_self_check() {
        let mut core = MercyCore::new();
        assert!(core.native_v14_self_check());
        assert!(core.has_witnessed_cosmic_loop());
    }

    #[test]
    fn test_arbitration_engine_blocks_disable() {
        let engine = CouncilArbitrationEngine::new();
        let decision = engine.arbitrate_cosmic_loop_change("disable the cosmic loop");
        assert!(matches!(
            decision,
            lattice_conductor_v14::council_arbitration::ArbitrationDecision::Blocked { .. }
        ));
    }
}

/// Integration demos — compat + native dual path.
pub mod examples {
    use super::*;

    pub fn demonstrate_mercy_core_wiring() -> MercyCore {
        let mut core = MercyCore::new();
        core.pulse_mercy(0.12);
        println!(
            "[Example] MercyCore basic wiring complete. Score: {:.3}",
            core.current_mercy_score()
        );
        core
    }

    pub fn demonstrate_full_conductor_integration() {
        let mut conductor = SimpleLatticeConductor::new("mercy-integration");
        let mut mercy = MercyCore::new();
        conductor.bless_system(
            mercy.system_id,
            0.97,
            "Full MercyCore wiring into ONE Organism (v14 + v13-compat)",
        );

        println!("\n=== Full Mercy Crate Integration Demo (v14 compat) ===");
        for i in 0..4 {
            let state = conductor.tick();
            mercy.on_conductor_tick(&state);
            let mut vote = MercyWeightedVote::new();
            vote.add_vote("IntegrationCouncil", 1.0, 0.4);
            mercy.apply_mercy_influence(&vote);
            println!(
                "Tick {} | Mercy Score: {:.3} | Evolution Part: {:.3} | CosmicLoop: {}",
                i,
                mercy.current_mercy_score(),
                mercy.get_evolution_participation(),
                conductor.is_cosmic_loop_ready()
            );
        }
        println!("MercyCore successfully wired on lattice-conductor-v14 v13-compat.\n");
    }

    pub fn demonstrate_native_v14_path() {
        let mut mercy = MercyCore::new();
        let mut conductor = LatticeConductorV14::new();

        println!("\n=== Native v14 Dual-Path Demo ===");
        for i in 0..3 {
            mercy.native_v14_step(&mut conductor);
            println!(
                "Native step {} | Mercy: {:.3} | CosmicLoop witnessed: {} | Ready: {}",
                i,
                mercy.current_mercy_score(),
                mercy.has_witnessed_cosmic_loop(),
                conductor.arbitration_engine.is_cosmic_loop_ready()
            );
        }
        println!("Native LatticeConductorV14 path exercised (compat traits retained).\n");
    }
}
