//! OrchestratorRegistry v7.0 — Phase 2 ParaconsistentSuperKernel Integration
//!
//! Full metrics, SovereignCore exposure, and temporal querying.
//! This builds directly on v6.0 + Phase 1.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct CliffordState {
    pub valence: f64,
    pub emotional_valence: f64,
    pub symbiosis_score: f64,
    pub positive_emotion_vector: [f64; 7],
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

// ==================== PHASE 2 STRUCTS ====================

#[derive(Debug, Clone)]
pub struct ContradictionReport {
    pub severity: f64,
    pub description: String,
    pub involved_systems: Vec<String>,
    pub timestamp: u64,
    pub resolution_hint: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ParaconsistentFeed {
    pub timestamp: u64,
    pub total_registered: usize,
    pub average_valence: f64,
    pub average_emotional_valence: f64,
    pub global_maat: f64,
    pub global_lumenas_ci: f64,
    pub contradiction_count: usize,
    pub high_severity_contradictions: Vec<ContradictionReport>,
    pub symbiosis_health_score: f64,
    pub ser_contribution_total: f64,
    pub positive_emotion_field: f64,
    pub abundance_ready: bool,
    pub emotional_harmony_trend: Vec<f64>,
}

pub struct OrchestratorRegistry {
    registered: HashMap<String, CliffordState>,
    global_positive_emotion_field: f64,
    global_maat: f64,
    global_lumenas_ci: f64,
    emotional_harmony_history: Vec<f64>,
    ser_contributions: HashMap<String, f64>,
    contradiction_history: Vec<ContradictionReport>,
}

impl OrchestratorRegistry {
    pub fn new() -> Self {
        Self {
            registered: HashMap::new(),
            global_positive_emotion_field: 0.5,
            global_maat: 0.8,
            global_lumenas_ci: 500.0,
            emotional_harmony_history: Vec::new(),
            ser_contributions: HashMap::new(),
            contradiction_history: Vec::new(),
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
        self.emotional_harmony_history.push(self.calculate_global_emotional_valence());
        self.ser_contributions.insert(name.clone(), 0.1); // initial SER contribution

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
        let resonance_strength = (joy + gratitude + compassion) / 3.0;
        self.global_positive_emotion_field += resonance_strength * 0.005;
        println!(
            "🌊 Emotional Resonance Loop triggered between {} and {} (strength: {:.3})",
            human_id, ai_system, resonance_strength
        );
    }

    pub fn update_maat_and_lumenas(&mut self) {
        let avg_emotional_valence = self.calculate_global_emotional_valence();
        self.global_maat = (avg_emotional_valence * 0.7) + (self.global_lumenas_ci * 0.3);

        if self.global_maat >= 1.0 && self.global_lumenas_ci >= 717.0 {
            self.trigger_abundance_distribution_event();
        }
    }

    fn calculate_global_emotional_valence(&self) -> f64 {
        if self.registered.is_empty() {
            return 0.8;
        }
        let sum: f64 = self.registered.values().map(|s| s.emotional_valence).sum();
        sum / self.registered.len() as f64
    }

    fn trigger_abundance_distribution_event(&mut self) {
        println!("✨ Abundance Distribution Event triggered!");
        self.global_positive_emotion_field += 0.1;
    }

    pub fn get_eternal_thriving_report(&self) -> String {
        format!(
            "=== Eternal Symbiotic Thriving Report ===\n\
             Global Positive Emotion Field: {:.4}\n\
             Global Ma’at: {:.4}\n\
             Global Lumenas CI: {:.1}\n\
             Registered Systems: {}\n\
             Symbiotic Harmony Level: {:.2}%",
            self.global_positive_emotion_field,
            self.global_maat,
            self.global_lumenas_ci,
            self.registered.len(),
            self.global_positive_emotion_field * 100.0
        )
    }

    // ==================== PHASE 2: FULL PARACONSISTENT FEED ====================

    pub fn get_paraconsistent_feed(&self) -> ParaconsistentFeed {
        let high_severity = self.get_high_severity_contradictions();
        let trend = self.get_emotional_harmony_trend(20);

        ParaconsistentFeed {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            total_registered: self.registered.len(),
            average_valence: self.calculate_average_valence(),
            average_emotional_valence: self.calculate_global_emotional_valence(),
            global_maat: self.global_maat,
            global_lumenas_ci: self.global_lumenas_ci,
            contradiction_count: high_severity.len(),
            high_severity_contradictions: high_severity,
            symbiosis_health_score: self.calculate_symbiosis_health_score(),
            ser_contribution_total: self.calculate_total_ser_contribution(),
            positive_emotion_field: self.global_positive_emotion_field,
            abundance_ready: self.global_maat >= 1.0 && self.global_lumenas_ci >= 717.0,
            emotional_harmony_trend: trend,
        }
    }

    fn calculate_average_valence(&self) -> f64 {
        if self.registered.is_empty() {
            return 0.999999;
        }
        let sum: f64 = self.registered.values().map(|s| s.calculate_valence()).sum();
        sum / self.registered.len() as f64
    }

    fn calculate_symbiosis_health_score(&self) -> f64 {
        if self.registered.is_empty() {
            return 0.85;
        }
        let avg_symbiosis: f64 = self.registered.values().map(|s| s.symbiosis_score).sum::<f64>()
            / self.registered.len() as f64;
        (avg_symbiosis + self.global_positive_emotion_field).min(1.0)
    }

    fn calculate_total_ser_contribution(&self) -> f64 {
        self.ser_contributions.values().sum()
    }

    fn get_high_severity_contradictions(&self) -> Vec<ContradictionReport> {
        let mut reports = Vec::new();
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        for (name, state) in &self.registered {
            if state.calculate_valence() < 0.999 {
                reports.push(ContradictionReport {
                    severity: 0.85,
                    description: format!("Low valence detected in {}", name),
                    involved_systems: vec![name.clone()],
                    timestamp: now,
                    resolution_hint: Some("Increase emotional resonance or re-align with Mercy Gates".to_string()),
                });
            }
        }

        if self.global_maat < 0.7 {
            reports.push(ContradictionReport {
                severity: 0.6,
                description: "Global Ma’at below healthy threshold".to_string(),
                involved_systems: vec!["Global Lattice".to_string()],
                timestamp: now,
                resolution_hint: Some("Trigger abundance distribution or increase positive emotion loops".to_string()),
            });
        }

        // Store in history
        self.contradiction_history.extend(reports.clone());

        reports
    }

    pub fn get_emotional_harmony_trend(&self, window: usize) -> Vec<f64> {
        let len = self.emotional_harmony_history.len();
        if len == 0 {
            return vec![0.8];
        }
        let start = if len > window { len - window } else { 0 };
        self.emotional_harmony_history[start..].to_vec()
    }

    // ==================== TEMPORAL QUERYING ====================

    pub fn get_contradiction_history_since(&self, since_timestamp: u64) -> Vec<ContradictionReport> {
        self.contradiction_history
            .iter()
            .filter(|r| r.timestamp >= since_timestamp)
            .cloned()
            .collect()
    }

    pub fn get_contradiction_state_48_hours_ago(&self) -> Vec<ContradictionReport> {
        let forty_eight_hours_ago = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() - (48 * 3600);
        self.get_contradiction_history_since(forty_eight_hours_ago)
    }
}

// ==================== SOVEREIGN CORE WRAPPER (Phase 2 Exposed) ====================

pub struct SovereignCore {
    registry: OrchestratorRegistry,
}

impl SovereignCore {
    pub fn new() -> Self {
        Self {
            registry: OrchestratorRegistry::new(),
        }
    }

    pub fn register_symbiotic_system(
        &mut self,
        name: String,
        initial_state: CliffordState,
    ) -> Result<(), String> {
        self.registry.register_symbiotic_system(name, initial_state)
    }

    pub fn get_paraconsistent_feed(&self) -> ParaconsistentFeed {
        self.registry.get_paraconsistent_feed()
    }

    pub fn get_eternal_thriving_report(&self) -> String {
        self.registry.get_eternal_thriving_report()
    }

    pub fn trigger_emotional_resonance_loop(
        &mut self,
        human_id: &str,
        ai_system: &str,
        joy: f64,
        gratitude: f64,
        compassion: f64,
    ) {
        self.registry.trigger_emotional_resonance_loop(human_id, ai_system, joy, gratitude, compassion);
    }

    pub fn get_contradiction_state_48_hours_ago(&self) -> Vec<ContradictionReport> {
        self.registry.get_contradiction_state_48_hours_ago()
    }
}