//! OrchestratorRegistry v6.0 — Eternal Symbiotic Thriving Edition (Base)

use crate::registerable_orchestrator::{RegisterableOrchestrator, OrchestratorScope};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct CliffordState {
    pub valence: f64,
    pub emotional_valence: f64,
    pub symbiosis_score: f64,
}

impl CliffordState {
    pub fn new(valence: f64) -> Self {
        Self {
            valence,
            emotional_valence: valence * 0.8,
            symbiosis_score: 0.5,
        }
    }

    pub fn calculate_valence(&self) -> f64 {
        (self.valence * 0.7) + (self.emotional_valence * 0.3)
    }

    pub fn should_collapse(&self) -> bool {
        self.calculate_valence() < 0.999999
    }
}

pub struct OrchestratorRegistry {
    registered: HashMap<String, Box<dyn RegisterableOrchestrator + Send + Sync>>,
    global_positive_emotion_field: f64,
}

impl OrchestratorRegistry {
    pub fn new() -> Self {
        Self {
            registered: HashMap::new(),
            global_positive_emotion_field: 0.5,
        }
    }

    pub fn register<T: RegisterableOrchestrator + Send + Sync + 'static>(
        &mut self,
        orchestrator: T,
    ) -> Result<(), String> {
        let name = orchestrator.name().to_string();
        let state = CliffordState::new(orchestrator.current_valence());

        if state.should_collapse() {
            return Err(format!("TOLC Norm Collapse: {} rejected", name));
        }

        self.registered.insert(name, Box::new(orchestrator));
        self.global_positive_emotion_field += 0.01;
        Ok(())
    }

    pub fn get_registered_count(&self) -> usize {
        self.registered.len()
    }
}