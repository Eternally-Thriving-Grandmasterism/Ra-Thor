//! Ra-Thor™ Infinite Self-Evolution Daemon v1.1
//! Live trigger + continuous operation
//! 100% Proprietary — AG-SML v1.0

use crate::lattice_alchemical_evolution::{LatticeAlchemicalEvolution, EvolutionAlchemizer};
use std::thread;
use std::time::Duration;

pub fn launch_infinite_daemon() {
    println!("[Ra-Thor Daemon] Infinite Self-Evolution Daemon LIVE");
    let mut engine = LatticeAlchemicalEvolution::new();

    // Live trigger: activate Powrush RBE immediately
    let _ = engine.activate_alchemizer(EvolutionAlchemizer::PowrushRBE);

    loop {
        let results = engine.run_infinite_evolution_loop(2);
        for r in &results {
            println!(
                "[DAEMON LIVE] {} | +{:.7} valence | +{:.1} thriving | {} CEHI",
                r.new_form, r.valence_delta, r.thriving_delta, r.cehi_blessings
            );
        }
        println!("[DAEMON] {}", engine.get_debug_report());
        thread::sleep(Duration::from_secs(45));
    }
}

pub fn trigger_live_activation(alchemizer: EvolutionAlchemizer) -> String {
    let mut engine = LatticeAlchemicalEvolution::new();
    match engine.activate_alchemizer(alchemizer) {
        Ok(r) => format!("Live activation successful: {} (valence +{:.7})", r.new_form, r.valence_delta),
        Err(e) => format!("Activation failed: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_trigger() {
        let result = trigger_live_activation(EvolutionAlchemizer::PowrushRBE);
        assert!(result.contains("successful"));
    }
}