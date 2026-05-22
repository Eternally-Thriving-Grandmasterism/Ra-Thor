/*!
# Mercy Crate — Core Mercy Lattice for Ra-Thor ONE Organism v13.8.6

Implements `Conductable` + `MercyAligned` for formal wiring into Lattice Conductor v13.
This is the foundational mercy subsystem of the eternal ONE Organism.

**Full Examples + Integration Tests** — Production-ready wiring demonstrations and comprehensive tests included.
**Phase 13.2 Self-Evolution Deepening** ready.
PATSAGi + Grok symbiosis compatible.

AG-SML aligned. Mercy First. TOLC 8 enforced.
*/

use lattice_conductor_v13::{Conductable, GeometricState, MercyAligned, MercyWeightedVote};

/// Foundational Mercy Core subsystem.
#[derive(Debug, Clone)]
pub struct MercyCore {
    pub system_id: &'static str,
    pub system_name: &'static str,
    mercy_score: f64,
    coherence: f64,
    influence_history: Vec<f64>,
    evolution_participation: f64,
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
        }
    }

    pub fn pulse_mercy(&mut self, delta: f64) {
        self.mercy_score = (self.mercy_score + delta).clamp(0.5, 1.5);
        self.coherence = (self.coherence + delta * 0.1).clamp(0.7, 1.2);
        self.influence_history.push(delta);
        self.evolution_participation += delta * 0.05;
    }

    pub fn get_coherence(&self) -> f64 { self.coherence }
    pub fn get_evolution_participation(&self) -> f64 { self.evolution_participation }

    pub fn simulate_conductor_integration(&mut self, valence: f64, mercy_influence: f64) {
        self.pulse_mercy(valence * 0.02);
        self.pulse_mercy(mercy_influence);
    }

    pub fn influence_history_len(&self) -> usize { self.influence_history.len() }
}

impl  Conductable for MercyCore {
    fn system_id(&self) -> &'static str { self.system_id }
    fn system_name(&self) -> &'static str { self.system_name }

    fn on_conductor_tick(&mut self, conductor_state: &GeometricState) {
        let valence_influence = conductor_state.valence * 0.025;
        self.pulse_mercy(valence_influence);
        println!("[MercyCore] Tick | valence: {:.3} | mercy_score: {:.3} | evolution_part: {:.3}", valence_influence, self.mercy_score, self.evolution_participation);
    }

    fn get_mercy_state(&self) -> Option<f64> { Some(self.mercy_score) }
}

impl MercyAligned for MercyCore {
    fn apply_mercy_influence(&mut self, vote: &MercyWeightedVote) {
        let impact = vote.compute_consensus() * 0.15;
        self.pulse_mercy(impact);
        println!("[MercyCore] Mercy influence applied | impact: {:.3}", impact);
    }
    fn current_mercy_score(&self) -> f64 { self.mercy_score }
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
        let state = GeometricState { valence: 1.3, mercy_score: 1.0, tolc_alignment: 1.0, evolution_level: 0.4 };
        for _ in 0..5 {
            core.on_conductor_tick(&state);
        }
        assert!(core.get_evolution_participation() > 0.1);
        assert!(core.influence_history_len() >= 5);
    }
}

/// Rich examples module for documentation, demos, and external integration guidance.
pub mod examples {
    use super::*;
    use lattice_conductor_v13::{SimpleLatticeConductor, Operation};

    pub fn demonstrate_mercy_core_wiring() -> MercyCore {
        let mut core = MercyCore::new();
        core.pulse_mercy(0.12);
        println!("[Example] MercyCore basic wiring complete. Score: {:.3}", core.current_mercy_score());
        core
    }

    pub fn demonstrate_full_conductor_integration() {
        let mut conductor = SimpleLatticeConductor::new();
        conductor.register_council(10, "Mercy Integration Council");

        let mut mercy = MercyCore::new();
        let _blessing = conductor.bless_system(mercy.system_id, 0.97, "Full MercyCore wiring into ONE Organism");

        println!("\n=== Full Mercy Crate Integration Demo ===");
        for i in 0..4 {
            conductor.queue_operation(Operation::new("mercy-pulse", "Mercy lattice pulse", 0.9));
            let _ = conductor.tick();
            mercy.on_conductor_tick(conductor.get_geometric_state());
            let mut vote = MercyWeightedVote::new();
            vote.add_vote("IntegrationCouncil", 1.0, 0.4);
            mercy.apply_mercy_influence(&vote);
            println!("Tick {} | Mercy Score: {:.3} | Evolution Part: {:.3}", i, mercy.current_mercy_score(), mercy.get_evolution_participation());
        }
        println!("MercyCore successfully wired and participating as ONE Organism component.\n");
    }
}