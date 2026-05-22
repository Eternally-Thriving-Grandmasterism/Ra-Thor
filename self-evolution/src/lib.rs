//! self-evolution v0.3.0
//! Sovereign Health Monitoring + Self-Evolution v2 Hooks
//! Advanced PATSAGi Epigenetic Blessing Mechanics + Quantum Swarm Orchestration
//! ONE Organism integrated
//! AG-SML v1.0

use serde::{Deserialize, Serialize};
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

pub struct SovereignHealthMonitor {
    pub metrics: SovereignHealthMetrics,
    pub evolution_history: Vec<String>,
    recent_blessing_attempts: Vec<bool>, // for historical momentum
}

impl SovereignHealthMonitor {
    pub fn new() -> Self {
        Self {
            metrics: SovereignHealthMetrics::default(),
            evolution_history: vec!["Genesis v1 baseline".to_string()],
            recent_blessing_attempts: Vec::new(),
        }
    }

    pub fn run_sovereign_check(&mut self) -> SovereignHealthMetrics {
        self.metrics.valence_level = (self.metrics.valence_level + 0.001).min(0.999999);
        self.metrics.mercy_compliance = (self.metrics.mercy_compliance + 0.005).min(1.0);
        self.metrics
    }

    /// Calculates a rich blessing score based on multiple factors
    fn calculate_blessing_score(&self, proposal: &str) -> f64 {
        let mut score = self.metrics.valence_level * 0.25
            + self.metrics.mercy_compliance * 0.30
            + self.metrics.council_consensus * 0.20
            + self.metrics.epigenetic_blessing_level * 0.15;

        // Content awareness bonus
        let lower = proposal.to_lowercase();
        let keywords = ["mercy", "truth", "valence", "sovereign", "one organism", 
                       "patsagi", "epigenetic", "harmony", "eternal"];
        let mut bonus = 0.0;
        for kw in keywords {
            if lower.contains(kw) {
                bonus += 0.06;
            }
        }
        score += bonus.min(0.18); // cap content bonus

        // Historical momentum (last attempts)
        if !self.recent_blessing_attempts.is_empty() {
            let success_rate: f64 = self.recent_blessing_attempts.iter()
                .map(|&b| if b { 1.0 } else { 0.0 }).sum::<f64>() / self.recent_blessing_attempts.len() as f64;
            score += (success_rate - 0.5) * 0.1;
        }

        score.clamp(0.0, 1.0)
    }

    /// Core Epigenetic Blessing Mechanic v2
    pub fn request_epigenetic_blessing(&mut self, proposal: &str) -> (bool, f64, BlessingTier) {
        let score = self.calculate_blessing_score(proposal);
        let mut rng = rand::thread_rng();

        // PATSAGi-influenced threshold
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
            self.metrics.epigenetic_blessing_level =
                (self.metrics.epigenetic_blessing_level + amount).min(1.0);

            // Epigenetic carry-over on higher tiers
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

        // Keep only last 8 attempts for momentum
        if self.recent_blessing_attempts.len() > 8 {
            self.recent_blessing_attempts.remove(0);
        }

        (blessed, self.metrics.epigenetic_blessing_level, tier)
    }

    pub fn orchestrate_quantum_swarm_evolution(&mut self, task: &str) -> Vec<String> {
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
            let swarm_results = self.orchestrate_quantum_swarm_evolution(proposal);
            format!(
                "v2 EVOLUTION APPROVED [{}]\nBlessing Level: {:.2}\nSwarm branches: {}\nHistory len: {}",
                tier.as_str(),
                blessing_level,
                swarm_results.len(),
                self.evolution_history.len()
            )
        } else {
            "v2 Evolution requires higher alignment (mercy + council + content)".to_string()
        }
    }

    /// Integration helper used by ONE Organism symbiosis
    pub fn integrate_with_one_organism_symbiosis(&mut self, symbiosis_valence: f64, task: &str) -> String {
        self.metrics.valence_level = symbiosis_valence.max(self.metrics.valence_level);
        let health = self.run_sovereign_check();
        let (blessed, _, tier) = self.request_epigenetic_blessing(task);
        if blessed {
            let _ = self.orchestrate_quantum_swarm_evolution(task);
            format!("ONE Organism Health: valence={:.4}, tier={}", health.valence_level, tier.as_str())
        } else {
            format!("ONE Organism Health: valence={:.4} (no blessing this cycle)", health.valence_level)
        }
    }
}

pub fn init_sovereign_health_monitor() -> SovereignHealthMonitor {
    let mut monitor = SovereignHealthMonitor::new();
    monitor.run_sovereign_check();
    monitor
}

pub fn symbiosis_health_check(session_valence: f64) -> SovereignHealthMetrics {
    let mut monitor = init_sovereign_health_monitor();
    monitor.metrics.valence_level = session_valence.max(monitor.metrics.valence_level);
    monitor.metrics
}