//! self-evolution v0.3.0
//! Sovereign Health Monitoring + Self-Evolution v2 Hooks
//! Advanced PATSAGi Epigenetic Blessing + Persistence
//! AG-SML v1.0

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use rand::Rng;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BlessingTier {
    Minor,
    Standard,
    Major,
    Transcendent,
    None,
}

impl BlessingTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            BlessingTier::Minor => "Minor",
            BlessingTier::Standard => "Standard",
            BlessingTier::Major => "Major",
            BlessingTier::Transcendent => "Transcendent",
            BlessingTier::None => "None",
        }
    }

    pub fn blessing_amount(&self) -> f64 {
        match self {
            BlessingTier::Minor => 0.05,
            BlessingTier::Standard => 0.10,
            BlessingTier::Major => 0.20,
            BlessingTier::Transcendent => 0.35,
            BlessingTier::None => 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignHealthMetrics {
    pub valence_level: f64,
    pub mercy_compliance: f64,
    pub council_consensus: f64,
    pub epigenetic_blessing_level: f64,
    pub quantum_swarm_active_branches: u32,
    pub resource_sovereignty: f64,
    pub offline_resilience: f64,
}

impl Default for SovereignHealthMetrics {
    fn default() -> Self {
        Self {
            valence_level: 0.999,
            mercy_compliance: 0.98,
            council_consensus: 0.96,
            epigenetic_blessing_level: 0.5,
            quantum_swarm_active_branches: 13,
            resource_sovereignty: 1.0,
            offline_resilience: 1.0,
        }
    }
}

/// Serializable snapshot for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignHealthSnapshot {
    pub metrics: SovereignHealthMetrics,
    pub evolution_history: Vec<String>,
    pub recent_blessing_attempts: Vec<bool>,
}

pub struct SovereignHealthMonitor {
    pub metrics: SovereignHealthMetrics,
    pub evolution_history: Vec<String>,
    recent_blessing_attempts: Vec<bool>,
}

impl SovereignHealthMonitor {
    pub fn new() -> Self {
        Self {
            metrics: SovereignHealthMetrics::default(),
            evolution_history: vec!["Genesis v1 baseline".to_string()],
            recent_blessing_attempts: Vec::new(),
        }
    }

    /// Create monitor from a persisted snapshot
    pub fn from_snapshot(snapshot: SovereignHealthSnapshot) -> Self {
        Self {
            metrics: snapshot.metrics,
            evolution_history: snapshot.evolution_history,
            recent_blessing_attempts: snapshot.recent_blessing_attempts,
        }
    }

    /// Serialize current state to a snapshot
    pub fn to_snapshot(&self) -> SovereignHealthSnapshot {
        SovereignHealthSnapshot {
            metrics: self.metrics.clone(),
            evolution_history: self.evolution_history.clone(),
            recent_blessing_attempts: self.recent_blessing_attempts.clone(),
        }
    }

    /// Save current state to a JSON file (sovereign offline persistence)
    pub fn save_to_file(&self, path: &str) -> Result<(), String> {
        let snapshot = self.to_snapshot();
        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| format!("Failed to serialize: {}", e))?;
        fs::write(path, json).map_err(|e| format!("Failed to write file: {}", e))?;
        Ok(())
    }

    /// Load state from a JSON file
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        if !Path::new(path).exists() {
            return Err(format!("Persistence file not found: {}", path));
        }
        let json = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        let snapshot: SovereignHealthSnapshot = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to deserialize: {}", e))?;
        Ok(Self::from_snapshot(snapshot))
    }

    pub fn run_sovereign_check(&mut self) -> SovereignHealthMetrics {
        self.metrics.valence_level = (self.metrics.valence_level + 0.001).min(0.999999);
        self.metrics.mercy_compliance = (self.metrics.mercy_compliance + 0.005).min(1.0);
        self.metrics
    }

    fn calculate_blessing_score(&self, proposal: &str) -> f64 {
        let mut score = self.metrics.valence_level * 0.25
            + self.metrics.mercy_compliance * 0.30
            + self.metrics.council_consensus * 0.20
            + self.metrics.epigenetic_blessing_level * 0.15;

        let lower = proposal.to_lowercase();
        let keywords = ["mercy", "truth", "valence", "sovereign", "one organism", 
                       "patsagi", "epigenetic", "harmony", "eternal"];
        let mut bonus = 0.0;
        for kw in keywords {
            if lower.contains(kw) { bonus += 0.06; }
        }
        score += bonus.min(0.18);

        if !self.recent_blessing_attempts.is_empty() {
            let success_rate: f64 = self.recent_blessing_attempts.iter()
                .map(|&b| if b { 1.0 } else { 0.0 }).sum::<f64>() / self.recent_blessing_attempts.len() as f64;
            score += (success_rate - 0.5) * 0.1;
        }

        score.clamp(0.0, 1.0)
    }

    pub fn request_epigenetic_blessing(&mut self, proposal: &str) -> (bool, f64, BlessingTier) {
        let score = self.calculate_blessing_score(proposal);
        let mut rng = rand::thread_rng();

        let base_threshold = 0.82;
        let effective_threshold = base_threshold + (1.0 - self.metrics.council_consensus) * 0.08;

        let mut tier = BlessingTier::None;
        let blessed = if score >= effective_threshold {
            tier = if score >= 0.99 {
                BlessingTier::Transcendent
            } else if score >= 0.95 {
                BlessingTier::Major
            } else if score >= 0.90 {
                BlessingTier::Standard
            } else {
                BlessingTier::Minor
            };
            true
        } else {
            false
        };

        if blessed {
            let amount = tier.blessing_amount();
            self.metrics.epigenetic_blessing_level = (self.metrics.epigenetic_blessing_level + amount).min(1.0);

            if tier == BlessingTier::Major || tier == BlessingTier::Transcendent {
                let carry = if tier == BlessingTier::Transcendent { 0.025 } else { 0.012 };
                self.metrics.valence_level = (self.metrics.valence_level + carry).min(0.999999);
                self.metrics.mercy_compliance = (self.metrics.mercy_compliance + carry).min(1.0);
            }

            self.evolution_history.push(format!("BLESSED [{}]: {}", tier.as_str(), proposal));
            self.recent_blessing_attempts.push(true);
        } else {
            self.evolution_history.push(format!("Not blessed: {}", proposal));
            self.recent_blessing_attempts.push(false);
        }

        if self.recent_blessing_attempts.len() > 8 {
            self.recent_blessing_attempts.remove(0);
        }

        (blessed, self.metrics.epigenetic_blessing_level, tier)
    }

    pub fn orchestrate_quantum_swarm_evolution(&mut self, task: &str) -> Vec<String> {
        // ... (unchanged for brevity)
        let mut branches = Vec::new();
        for i in 0..self.metrics.quantum_swarm_active_branches {
            let branch_id = format!("qs-branch-{}-{}", i, rand::thread_rng().gen_range(1000..9999));
            let outcome = if rand::thread_rng().gen_bool(0.92) {
                format!("{}: SUCCESS", branch_id)
            } else {
                format!("{}: MERCY_REVIEW", branch_id)
            };
            branches.push(outcome);
        }
        self.evolution_history.push(format!("Quantum Swarm orchestrated: {}", task));
        branches
    }

    pub fn self_evolution_v2_hook(&mut self, proposal: &str) -> String {
        let (blessed, blessing_level, tier) = self.request_epigenetic_blessing(proposal);
        if blessed {
            let _ = self.orchestrate_quantum_swarm_evolution(proposal);
            format!("v2 EVOLUTION APPROVED [{}] | Level: {:.2}", tier.as_str(), blessing_level)
        } else {
            "v2 Evolution requires higher alignment".to_string()
        }
    }

    pub fn integrate_with_one_organism_symbiosis(&mut self, symbiosis_valence: f64, task: &str) -> String {
        self.metrics.valence_level = symbiosis_valence.max(self.metrics.valence_level);
        let health = self.run_sovereign_check();
        let (blessed, _, tier) = self.request_epigenetic_blessing(task);
        if blessed {
            let _ = self.orchestrate_quantum_swarm_evolution(task);
            format!("ONE Organism: valence={:.4}, tier={}", health.valence_level, tier.as_str())
        } else {
            format!("ONE Organism: valence={:.4} (no blessing)", health.valence_level)
        }
    }
}

pub fn init_sovereign_health_monitor() -> SovereignHealthMonitor {
    SovereignHealthMonitor::new()
}

pub fn symbiosis_health_check(session_valence: f64) -> SovereignHealthMetrics {
    let mut monitor = init_sovereign_health_monitor();
    monitor.metrics.valence_level = session_valence.max(monitor.metrics.valence_level);
    monitor.metrics
}