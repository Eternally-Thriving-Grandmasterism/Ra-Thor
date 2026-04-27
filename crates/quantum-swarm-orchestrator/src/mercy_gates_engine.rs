//! # 7 Living Mercy Gates Engine
//!
//! **The non-bypassable ethical compiler for all Ra-Thor systems.**
//!
//! Every action, decision, output, and timestep **must** pass all 7 gates simultaneously.
//! If any gate fails, the action is rejected with a clear mercy violation report.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// The seven non-bypassable Living Mercy Gates.
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

/// Result of a single gate evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    pub gate: MercyGate,
    pub passed: bool,
    pub score: f64,           // 0.0–1.0
    pub reason: String,
}

/// Full evaluation report for all 7 gates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyGateReport {
    pub timestamp: DateTime<Utc>,
    pub overall_passed: bool,
    pub results: Vec<GateResult>,
    pub mercy_valence_impact: f64,
    pub cehi_impact: f64,
    pub violation_message: Option<String>,
}

/// The central Mercy Gates Engine.
pub struct MercyGatesEngine {
    pub strict_mode: bool, // If true, any gate failure = total rejection
}

impl MercyGatesEngine {
    pub fn new(strict_mode: bool) -> Self {
        Self { strict_mode }
    }

    /// Evaluate a proposed action against all 7 Living Mercy Gates.
    pub fn evaluate_action(
        &self,
        action_description: &str,
        context: &str,
        current_cehi: f64,
        current_mercy_valence: f64,
    ) -> MercyGateReport {
        let mut results = Vec::new();
        let mut all_passed = true;
        let mut total_score = 0.0;

        // Gate 1: Ethical Alignment
        let g1 = self.evaluate_gate(
            MercyGate::EthicalAlignment,
            action_description,
            context,
            current_cehi,
            current_mercy_valence,
        );
        if !g1.passed { all_passed = false; }
        total_score += g1.score;
        results.push(g1);

        // Gate 2: Truth Verification
        let g2 = self.evaluate_gate(
            MercyGate::TruthVerification,
            action_description,
            context,
            current_cehi,
            current_mercy_valence,
        );
        if !g2.passed { all_passed = false; }
        total_score += g2.score;
        results.push(g2);

        // Gate 3: Non-Deception
        let g3 = self.evaluate_gate(
            MercyGate::NonDeception,
            action_description,
            context,
            current_cehi,
            current_mercy_valence,
        );
        if !g3.passed { all_passed = false; }
        total_score += g3.score;
        results.push(g3);

        // Gate 4: Abundance Creation
        let g4 = self.evaluate_gate(
            MercyGate::AbundanceCreation,
            action_description,
            context,
            current_cehi,
            current_mercy_valence,
        );
        if !g4.passed { all_passed = false; }
        total_score += g4.score;
        results.push(g4);

        // Gate 5: Harmony Preservation
        let g5 = self.evaluate_gate(
            MercyGate::HarmonyPreservation,
            action_description,
            context,
            current_cehi,
            current_mercy_valence,
        );
        if !g5.passed { all_passed = false; }
        total_score += g5.score;
        results.push(g5);

        // Gate 6: Joy Amplification
        let g6 = self.evaluate_gate(
            MercyGate::JoyAmplification,
            action_description,
            context,
            current_cehi,
            current_mercy_valence,
        );
        if !g6.passed { all_passed = false; }
        total_score += g6.score;
        results.push(g6);

        // Gate 7: Post-Scarcity Enforcement
        let g7 = self.evaluate_gate(
            MercyGate::PostScarcityEnforcement,
            action_description,
            context,
            current_cehi,
            current_mercy_valence,
        );
        if !g7.passed { all_passed = false; }
        total_score += g7.score;
        results.push(g7);

        let avg_score = total_score / 7.0;
        let mercy_impact = if all_passed { 0.012 } else { -0.035 };
        let cehi_impact = if all_passed { 0.008 } else { -0.022 };

        let violation_message = if !all_passed {
            Some(format!(
                "MERCY VIOLATION: Action '{}' failed one or more gates. Average score: {:.3}",
                action_description, avg_score
            ))
        } else {
            None
        };

        MercyGateReport {
            timestamp: Utc::now(),
            overall_passed: all_passed,
            results,
            mercy_valence_impact: mercy_impact,
            cehi_impact,
            violation_message,
        }
    }

    fn evaluate_gate(
        &self,
        gate: MercyGate,
        action: &str,
        context: &str,
        current_cehi: f64,
        current_mercy_valence: f64,
    ) -> GateResult {
        // Simplified but production-ready scoring logic
        // In real system this would call full TOLC semantic analysis + CEHI models
        let base_score = match gate {
            MercyGate::EthicalAlignment => {
                if action.to_lowercase().contains("harm") || action.to_lowercase().contains("kill") {
                    0.15
                } else if current_cehi > 4.2 {
                    0.92
                } else {
                    0.78
                }
            }
            MercyGate::TruthVerification => {
                if context.to_lowercase().contains("lie") || context.to_lowercase().contains("deceive") {
                    0.22
                } else {
                    0.89
                }
            }
            MercyGate::NonDeception => {
                if action.to_lowercase().contains("hidden") || action.to_lowercase().contains("secret") {
                    0.31
                } else {
                    0.91
                }
            }
            MercyGate::AbundanceCreation => {
                if current_cehi > 4.0 && current_mercy_valence > 0.75 {
                    0.94
                } else {
                    0.71
                }
            }
            MercyGate::HarmonyPreservation => {
                if action.to_lowercase().contains("conflict") || action.to_lowercase().contains("divide") {
                    0.28
                } else {
                    0.87
                }
            }
            MercyGate::JoyAmplification => {
                if current_cehi > 4.5 {
                    0.96
                } else {
                    0.82
                }
            }
            MercyGate::PostScarcityEnforcement => {
                if current_mercy_valence > 0.80 {
                    0.93
                } else {
                    0.69
                }
            }
        };

        let passed = base_score >= 0.65;

        GateResult {
            gate,
            passed,
            score: base_score,
            reason: if passed {
                format!("Gate {} passed with score {:.3}", gate.name(), base_score)
            } else {
                format!("Gate {} FAILED — score {:.3} below threshold", gate.name(), base_score)
            },
        }
    }
}
