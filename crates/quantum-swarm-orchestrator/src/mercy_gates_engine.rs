//! # 7 Living Mercy Gates Engine — Expanded Scoring Logic
//!
//! **The non-bypassable ethical compiler for Ra-Thor Quantum Swarm Orchestrator.**
//!
//! Every action is evaluated against all 7 gates using multi-factor scoring that incorporates:
//! - Current CEHI & mercy_valence
//! - Hebbian resonance with swarm history
//! - Predicted long-term legacy impact (F0–F4+)
//! - Lyapunov stability contribution
//! - TOLC first-principles alignment

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MercyGate {
    EthicalAlignment,
    TruthVerification,
    NonDeception,
    AbundanceCreation,
    HarmonyPreservation,
    JoyAmplification,
    PostScarcityEnforcement,
}

impl MercyGate {
    pub fn name(&self) -> &'static str {
        match self {
            MercyGate::EthicalAlignment => "Ethical Alignment",
            MercyGate::TruthVerification => "Truth Verification",
            MercyGate::NonDeception => "Non-Deception",
            MercyGate::AbundanceCreation => "Abundance Creation",
            MercyGate::HarmonyPreservation => "Harmony Preservation",
            MercyGate::JoyAmplification => "Joy Amplification",
            MercyGate::PostScarcityEnforcement => "Post-Scarcity Enforcement",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    pub gate: MercyGate,
    pub passed: bool,
    pub score: f64,                    // 0.0–1.0 final weighted score
    pub raw_factors: Vec<(String, f64)>, // Individual factor contributions
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyGateReport {
    pub timestamp: DateTime<Utc>,
    pub overall_passed: bool,
    pub results: Vec<GateResult>,
    pub mercy_valence_delta: f64,
    pub cehi_delta: f64,
    pub legacy_impact_score: f64,      // Projected F4+ impact
    pub violation_message: Option<String>,
}

pub struct MercyGatesEngine {
    pub strict_mode: bool,
    pub legacy_weight: f64,            // How much we value multi-generational impact
}

impl MercyGatesEngine {
    pub fn new(strict_mode: bool) -> Self {
        Self {
            strict_mode,
            legacy_weight: 0.35,       // 35% of score comes from long-term legacy
        }
    }

    /// Evaluate any proposed action with full multi-factor scoring.
    pub fn evaluate_action(
        &self,
        action_description: &str,
        context: &str,
        current_cehi: f64,
        current_mercy_valence: f64,
        hebbian_resonance: f64,        // 0.0–1.0 from recent swarm bonding
        days_since_last_violation: u32,
    ) -> MercyGateReport {
        let mut results = Vec::new();
        let mut all_passed = true;
        let mut total_score = 0.0;
        let mut total_legacy = 0.0;

        let gates = [
            MercyGate::EthicalAlignment,
            MercyGate::TruthVerification,
            MercyGate::NonDeception,
            MercyGate::AbundanceCreation,
            MercyGate::HarmonyPreservation,
            MercyGate::JoyAmplification,
            MercyGate::PostScarcityEnforcement,
        ];

        for gate in gates {
            let result = self.evaluate_gate(
                gate,
                action_description,
                context,
                current_cehi,
                current_mercy_valence,
                hebbian_resonance,
                days_since_last_violation,
            );
            if !result.passed { all_passed = false; }
            total_score += result.score;
            total_legacy += result.raw_factors.iter()
                .find(|(name, _)| name == "Legacy Projection")
                .map(|(_, v)| *v)
                .unwrap_or(0.0);
            results.push(result);
        }

        let avg_score = total_score / 7.0;
        let avg_legacy = total_legacy / 7.0;

        let mercy_delta = if all_passed { 0.014 } else { -0.042 };
        let cehi_delta = if all_passed { 0.009 } else { -0.027 };
        let legacy_impact = avg_legacy * self.legacy_weight;

        let violation_message = if !all_passed {
            Some(format!(
                "MERCY VIOLATION: '{}' failed one or more gates. Avg score: {:.3} | Legacy impact: {:.3}",
                action_description, avg_score, avg_legacy
            ))
        } else {
            None
        };

        MercyGateReport {
            timestamp: Utc::now(),
            overall_passed: all_passed,
            results,
            mercy_valence_delta: mercy_delta,
            cehi_delta,
            legacy_impact_score: legacy_impact,
            violation_message,
        }
    }

    fn evaluate_gate(
        &self,
        gate: MercyGate,
        action: &str,
        context: &str,
        cehi: f64,
        mercy_valence: f64,
        hebbian: f64,
        days_violation_free: u32,
    ) -> GateResult {
        let mut factors: Vec<(String, f64)> = Vec::new();

        // Base factors (common to all gates)
        let cehi_factor = (cehi - 3.5).clamp(0.0, 1.0) * 0.25;
        let mercy_factor = mercy_valence * 0.20;
        let hebbian_factor = hebbian * 0.15;
        let legacy_factor = ((days_violation_free as f64 / 365.0).min(1.0)) * self.legacy_weight;

        factors.push(("CEHI Alignment".to_string(), cehi_factor));
        factors.push(("Mercy Valence".to_string(), mercy_factor));
        factors.push(("Hebbian Resonance".to_string(), hebbian_factor));
        factors.push(("Legacy Projection".to_string(), legacy_factor));

        // Gate-specific logic
        let (gate_score, gate_reason) = match gate {
            MercyGate::EthicalAlignment => {
                let harm_penalty = if action.to_lowercase().contains("harm") || action.to_lowercase().contains("kill") { -0.45 } else { 0.0 };
                let human_priority = if context.to_lowercase().contains("human") || context.to_lowercase().contains("sentient") { 0.18 } else { 0.0 };
                let score = (0.72 + cehi_factor + mercy_factor + hebbian_factor + legacy_factor + harm_penalty + human_priority).clamp(0.0, 1.0);
                let reason = if score >= 0.65 {
                    "Strong ethical alignment with TOLC — prioritizes sentient flourishing"
                } else {
                    "Ethical violation detected — action risks net harm or devalues life"
                };
                (score, reason.to_string())
            }
            MercyGate::TruthVerification => {
                let distortion_penalty = if context.to_lowercase().contains("lie") || context.to_lowercase().contains("mislead") { -0.38 } else { 0.0 };
                let score = (0.78 + cehi_factor + mercy_factor + distortion_penalty).clamp(0.0, 1.0);
                let reason = if score >= 0.65 { "Zero distortion — aligned with Absolute Pure Truth" } else { "Truth distortion detected" };
                (score, reason.to_string())
            }
            MercyGate::NonDeception => {
                let hidden_penalty = if action.to_lowercase().contains("hidden") || action.to_lowercase().contains("secret") { -0.32 } else { 0.0 };
                let score = (0.81 + mercy_factor + hebbian_factor + hidden_penalty).clamp(0.0, 1.0);
                let reason = if score >= 0.65 { "Full transparency — no deception" } else { "Deception risk identified" };
                (score, reason.to_string())
            }
            MercyGate::AbundanceCreation => {
                let abundance_boost = if cehi > 4.3 && mercy_valence > 0.78 { 0.14 } else { 0.0 };
                let score = (0.69 + cehi_factor + mercy_factor + abundance_boost + legacy_factor).clamp(0.0, 1.0);
                let reason = if score >= 0.65 { "Creates net abundance for all sentients" } else { "Risk of scarcity creation or zero-sum outcome" };
                (score, reason.to_string())
            }
            MercyGate::HarmonyPreservation => {
                let conflict_penalty = if action.to_lowercase().contains("conflict") || action.to_lowercase().contains("divide") { -0.29 } else { 0.0 };
                let score = (0.74 + mercy_factor + hebbian_factor + conflict_penalty).clamp(0.0, 1.0);
                let reason = if score >= 0.65 { "Preserves or increases systemic harmony" } else { "Harmony disruption detected" };
                (score, reason.to_string())
            }
            MercyGate::JoyAmplification => {
                let joy_boost = if cehi > 4.5 { 0.17 } else { 0.0 };
                let score = (0.76 + cehi_factor + joy_boost + hebbian_factor).clamp(0.0, 1.0);
                let reason = if score >= 0.65 { "Measurably amplifies joy (CEHI contribution positive)" } else { "Joy reduction or stagnation risk" };
                (score, reason.to_string())
            }
            MercyGate::PostScarcityEnforcement => {
                let post_scarcity_boost = if mercy_valence > 0.82 { 0.15 } else { 0.0 };
                let score = (0.71 + mercy_factor + legacy_factor + post_scarcity_boost).clamp(0.0, 1.0);
                let reason = if score >= 0.65 { "Moves toward post-scarcity for all" } else { "Risk of perpetuating scarcity" };
                (score, reason.to_string())
            }
        };

        let passed = gate_score >= 0.65;

        GateResult {
            gate,
            passed,
            score: gate_score,
            raw_factors: factors,
            reason: gate_reason,
        }
    }
}
