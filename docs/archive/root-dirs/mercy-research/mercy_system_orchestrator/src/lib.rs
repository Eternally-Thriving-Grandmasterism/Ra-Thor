// mercy_system_orchestrator/src/lib.rs — Valence-Gated Propulsion Orchestrator
#[derive(Debug, Clone)]
pub enum PropulsionSource {
    Electric,
    BioJet,
    Hydrogen,
}

#[derive(Debug, Clone)]
pub struct FlightPhase {
    pub name: String,
    pub thrust_demand: f64,  // normalized 0.0–1.0
    pub efficiency_priority: f64,
}

pub struct MercyOrchestrator {
    pub valence: f64,
}

impl MercyOrchestrator {
    pub fn new() -> Self {
        MercyOrchestrator { valence: 1.0 }
    }

    pub fn select_source(&self, phase: &FlightPhase) -> PropulsionSource {
        if self.valence < 0.9999999 {
            println!("Mercy shield: Orchestration paused — valence {:.7}", self.valence);
            return PropulsionSource::Electric; // fallback
        }

        // Simple valence-gated logic (expandable to ML / real-time optimization)
        if phase.thrust_demand > 0.8 {
            PropulsionSource::BioJet // high-thrust takeoff/climb
        } else if phase.efficiency_priority > 0.7 {
            PropulsionSource::Electric // cruise / precision
        } else {
            PropulsionSource::Hydrogen // long-range extension
        }
    }
}

pub fn simulate_orchestration(phase_name: &str, thrust: f64, efficiency: f64) -> PropulsionSource {
    let orchestrator = MercyOrchestrator::new();
    let phase = FlightPhase {
        name: phase_name.to_string(),
        thrust_demand: thrust,
        efficiency_priority: efficiency,
    };
    let source = orchestrator.select_source(&phase);
    println!("Mercy-approved: {} phase — selected {:?}", phase_name, source);
    source
}
