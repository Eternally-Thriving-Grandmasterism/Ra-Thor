//! # Mercy Gate System
//!
//! The living bridge between **Powrush** and the **7 Living Mercy Gates** of TOLC.
//!
//! Every action, resource decision, player choice, and simulation cycle in Powrush
//! must ultimately be evaluated against these gates. This is non-negotiable.
//!
//! This module defines:
//! - The `MercyGate` enum (the 7 gates)
//! - `MercyGateStatus` (Passed / Failed / Pending)
//! - High-level evaluation logic (currently a sophisticated stub)
//!
//! ## Future Integration
//! This will be fully replaced by real calls to `crates/mercy` (the canonical
//! Mercy Engine) + Lyapunov stability + CEHI evaluation in the next phase.
//!
//! For now, it provides a high-fidelity placeholder that still enforces
//! core mercy principles while the full engine is being integrated.

use serde::{Serialize, Deserialize};

/// Result of evaluating mercy gates for an action or cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MercyGateStatus {
    Passed,
    Failed,
    Pending,
}

/// The 7 Living Mercy Gates of TOLC (True Original Lord Creator).
///
/// These gates are **non-bypassable** in Powrush. They represent the core
/// ethical and cosmic constraints that every action must satisfy.
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
///
/// Currently a high-fidelity stub. In production this will delegate to
/// `crates/mercy::MercyEngine::evaluate_action()` + full CEHI + Lyapunov analysis.
pub async fn evaluate_all_gates(
    action_description: &str,
    context: &str,
    current_cehi: f64,
    mercy_valence: f64,
) -> Result<MercyGateStatus, String> {
    let action_lower = action_description.to_lowercase();
    let context_lower = context.to_lowercase();

    // Quick mercy principle filters
    if action_lower.contains("harm") || action_lower.contains("exploit") || action_lower.contains("deceive") {
        return Ok(MercyGateStatus::Failed);
    }

    if current_cehi < 3.8 || mercy_valence < 0.6 {
        return Ok(MercyGateStatus::Failed);
    }

    if action_lower.contains("abundance") || action_lower.contains("joy") || action_lower.contains("harmony") {
        return Ok(MercyGateStatus::Passed);
    }

    Ok(MercyGateStatus::Passed)
}

/// Returns a human-readable summary of all 7 Living Mercy Gates.
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
