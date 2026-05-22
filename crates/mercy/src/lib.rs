/*!
# Mercy Crate — Core Mercy Lattice for Ra-Thor ONE Organism v13.8.3

Implements `Conductable` + `MercyAligned` for formal wiring into Lattice Conductor v13.
This is the foundational mercy subsystem of the eternal ONE Organism.

AG-SML aligned. Mercy First. TOLC 8 enforced.
*/

use lattice_conductor_v13::{Conductable, GeometricState, MercyAligned, MercyWeightedVote};

/// Foundational Mercy Core subsystem.
/// Can be blessed into the Lattice Conductor and participate in mercy-weighted coordination.
#[derive(Debug, Clone)]
pub struct MercyCore {
    pub system_id: &'static str,
    pub system_name: &'static str,
    mercy_score: f64,
    coherence: f64,
    influence_history: Vec<f64>,
}

impl MercyCore {
    pub fn new() -> Self {
        Self {
            system_id: "mercy-core",
            system_name: "Mercy Core Lattice",
            mercy_score: 0.98,
            coherence: 1.0,
            influence_history: Vec::new(),
        }
    }

    pub fn pulse_mercy(&mut self, delta: f64) {
        self.mercy_score = (self.mercy_score + delta).clamp(0.5, 1.5);
        self.coherence = (self.coherence + delta * 0.1).clamp(0.7, 1.2);
        self.influence_history.push(delta);
    }

    pub fn get_coherence(&self) -> f64 {
        self.coherence
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
        let valence_influence = conductor_state.valence * 0.02;
        self.pulse_mercy(valence_influence);
        println!(
            "[MercyCore] Tick received | valence_influence: {:.3} | mercy_score: {:.3} | coherence: {:.3}",
            valence_influence, self.mercy_score, self.coherence
        );
    }

    fn get_mercy_state(&self) -> Option<f64> {
        Some(self.mercy_score)
    }
}

impl MercyAligned for MercyCore {
    fn apply_mercy_influence(&mut self, vote: &MercyWeightedVote) {
        let impact = vote.compute_consensus() * 0.12;
        self.pulse_mercy(impact);
        println!(
            "[MercyCore] Mercy influence applied | impact: {:.3} | new mercy_score: {:.3}",
            impact, self.mercy_score
        );
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
    }
}
