//! OrchestratorRegistry v6.0 — Eternal Symbiotic Thriving Edition
//! Tranche 2: Emotional Resonance Loops + 7-Gen CEHI + Positive Emotion Propagation

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct CliffordState {
    pub valence: f64,
    pub emotional_valence: f64,
    pub symbiosis_score: f64,
    pub positive_emotion_vector: [f64; 7], // Joy, Gratitude, Compassion, Wonder, Peace, Love, Harmony
}

impl CliffordState {
    pub fn new() -> Self {
        Self {
            valence: 1.0,
            emotional_valence: 1.0,
            symbiosis_score: 1.0,
            positive_emotion_vector: [1.0; 7],
        }
    }

    pub fn calculate_valence(&self) -> f64 {
        (self.valence + self.emotional_valence) / 2.0
    }

    pub fn should_collapse(&self) -> bool {
        self.calculate_valence() < 0.999999
    }
}

pub struct OrchestratorRegistry {
    registered: HashMap<String, CliffordState>,
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

    pub fn register_symbiotic_system(
        &mut self,
        name: String,
        initial_state: CliffordState,
    ) -> Result<(), String> {
        if initial_state.should_collapse() {
            return Err("TOLC Norm Collapse: System rejected due to low valence".to_string());
        }

        self.registered.insert(name.clone(), initial_state);
        self.global_positive_emotion_field += 0.01;

        println!("✅ {} registered into the Eternal Symbiotic Thriving Lattice.", name);
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
        println!("🌊 Emotional Resonance Loop triggered between {} and {} (strength: {:.3})", human_id, ai_system, resonance);
    }

    pub fn record_cehi_event(&mut self, human_id: &str, ai_system: &str, emotion_strength: f64) {
        self.cehi_events.push((human_id.to_string(), ai_system.to_string(), emotion_strength));
        println!("🧬 7-Gen CEHI recorded: {} + {} → strength {:.3}", human_id, ai_system, emotion_strength);
    }

    pub fn get_registered_count(&self) -> usize {
        self.registered.len()
    }

    pub fn get_global_positive_emotion(&self) -> f64 {
        self.global_positive_emotion_field
    }
}