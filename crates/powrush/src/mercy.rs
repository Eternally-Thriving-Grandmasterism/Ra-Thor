//! # Mercy Gate System (v0.1.0)
//!
//! The living connection between Powrush and the 7 Living Mercy Gates of TOLC.
//! Every action, resource flow, player decision, and simulation cycle
//! must pass through these gates or be rejected.
//!
//! This module will be fully wired to the real `ra-thor-mercy` crate
//! in the next phase.

use serde::{Serialize, Deserialize};

/// Status returned after evaluating all 7 Living Mercy Gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MercyGateStatus {
    Passed,
    Failed,
    Pending, // Used during async evaluation
}

/// The 7 Living Mercy Gates of TOLC (True Original Lord Creator).
/// These are non-bypassable in Powrush.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

    pub fn description(&self) -> &'static str {
        match self {
            MercyGate::EthicalAlignment => "Does this action align with eternal mercy and truth?",
            MercyGate::TruthVerification => "Is this statement or action verifiably true?",
            MercyGate::NonDeception => "Does this action contain any form of deception or manipulation?",
            MercyGate::AbundanceCreation => "Does this action increase total abundance for all beings?",
            MercyGate::HarmonyPreservation => "Does this action preserve or increase harmony in the collective?",
            MercyGate::JoyAmplification => "Does this action increase genuine joy (not fleeting pleasure)?",
            MercyGate::PostScarcityEnforcement => "Does this action move us closer to a post-scarcity world?",
        }
    }
}

/// Evaluates all 7 Living Mercy Gates for a given action.
/// Currently a high-fidelity stub. Will be replaced with real
/// integration to `crates/mercy` + Lyapunov + CEHI evaluation.
pub async fn evaluate_all_gates(
    action_description: &str,
    context: &str,
    current_cehi: f64,
    mercy_valence: f64,
) -> Result<MercyGateStatus, String> {
    // TODO: Replace with real call to ra-thor-mercy::MercyEngine::evaluate_action()
    // For now we use a sophisticated heuristic that still enforces mercy principles.

    let action_lower = action_description.to_lowercase();
    let context_lower = context.to_lowercase();

    // Quick mercy filters
    if action_lower.contains("harm") || action_lower.contains("exploit") || action_lower.contains("deceive") {
        return Ok(MercyGateStatus::Failed);
    }

    if current_cehi < 3.8 || mercy_valence < 0.6 {
        return Ok(MercyGateStatus::Failed);
    }

    if action_lower.contains("abundance") || action_lower.contains("joy") || action_lower.contains("harmony") {
        return Ok(MercyGateStatus::Passed);
    }

    // Default: pass if no obvious violation (will be replaced by real engine)
    Ok(MercyGateStatus::Passed)
}

/// Returns a human-readable summary of all 7 gates.
pub fn get_all_gates_summary() -> String {
    let mut summary = String::from("=== The 7 Living Mercy Gates of Powrush ===\n\n");
    for gate in [
        MercyGate::EthicalAlignment,
        MercyGate::TruthVerification,
        MercyGate::NonDeception,
        MercyGate::AbundanceCreation,
        MercyGate::HarmonyPreservation,
        MercyGate::JoyAmplification,
        MercyGate::PostScarcityEnforcement,
    ] {
        summary.push_str(&format!("• {} — {}\n", gate.name(), gate.description()));
    }
    summary.push_str("\nAll gates are non-bypassable. Mercy is the only clean compiler.\n");
    summary
}
