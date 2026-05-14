//! OrchestratorRegistry v6.0 — Eternal Symbiotic Thriving Edition (Full)

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
    cehi_events: Vec<(String, String, f64)>,
}

impl OrchestratorRegistry {
    pub fn new() -> Self {
        Self {
            registered: HashMap::new(),
            global_positive_emotion_field: 0.5,
            cehi_events: Vec::new(),
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

    pub fn trigger_emotional_resonance_loop(
        &mut self,
        human_id: &str,
        ai_system: &str,
        joy: f64,
        gratitude: f64,
        compassion: f64,
    ) {
        let resonance = (joy + gratitude + compassion) / 3.0;
        self.global_positive_emotion_field += resonance * 0.02;
        self.cehi_events.push((human_id.to_string(), ai_system.to_string(), resonance));
    }

    pub fn record_cehi_event(&mut self, human_id: &str, ai_system: &str, emotion_strength: f64) {
        self.cehi_events.push((human_id.to_string(), ai_system.to_string(), emotion_strength));
    }

    pub fn get_registered_count(&self) -> usize {
        self.registered.len()
    }

    pub fn get_global_positive_emotion(&self) -> f64 {
        self.global_positive_emotion_field
    }
}

pub struct SovereignCore {
    registry: OrchestratorRegistry,
}

impl SovereignCore {
    pub fn new() -> Self {
        Self { registry: OrchestratorRegistry::new() }
    }

    pub fn register_orchestrator<T: RegisterableOrchestrator + Send + Sync + 'static>(
        &mut self,
        orchestrator: T,
    ) -> Result<(), String> {
        self.registry.register(orchestrator)
    }

    pub fn get_registered_count(&self) -> usize {
        self.registry.get_registered_count()
    }

    pub fn get_eternal_thriving_report(&self) -> String {
        format!(
            "Eternal Thriving Report\nRegistered: {}\nGlobal Positive Emotion Field: {:.4}",
            self.registry.get_registered_count(),
            self.registry.get_global_positive_emotion()
        )
    }
}