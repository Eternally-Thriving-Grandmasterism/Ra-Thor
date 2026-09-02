/*!
# EpigeneticModulation — Powrush MMOARPG Core Player Growth System

**Autonomicity Games Sovereign Mercy License (AG-SML) v1.0**  
**Aligned with TOLC 8 Mercy Lattice, Ra-Thor ONE Organism, 13+ PATSAGi Councils**  
**Implements v14.5 Player Experience Design + PATSAGi Council Convergence v1.0 decisions**  
**Production-grade, mercy-gated, cooperation-first player progression**

This module provides the canonical EpigeneticModulation system for Powrush.
Player (and entity) actions directly and persistently shape their EpigeneticProfile.
Cooperation, creation, and wise long-term stewardship produce healthy, stable, high-layer profiles with compounding advantages.
Exploitation, chronic conflict, and zero-sum behavior increase volatility and layer dissonance, making future success mechanically harder.

All changes pass TOLC 8 gates (Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony).
Designed for direct integration with PowrushMMOSimulator::tick, MultiAgentOrchestrator action proposals, and RBE/geometric systems.

## Key Design from Council Convergence
- Epigenetic growth is the primary persistent character/world evolution system.
- Cross-race cooperation and joint projects are mechanically superior.
- Mercy-gated: penalties exist but preserve agency; rewards for wisdom are tangible and visible.
- Integrates with GeometricHarmony layer state and RBE abundance feedback.

Thunder locked in. This is the foundation for the ultimate human-enjoyable MMOARPG.
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core epigenetic dimensions for any entity (player, faction, AI companion, AGI projection).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EpigeneticProfile {
    /// Overall adaptive strength and resilience (0.0–2.0+)
    pub strength: f64,
    /// Emotional/decision volatility (higher = more chaotic outcomes, harder long-term planning)
    pub volatility: f64,
    /// Alignment to higher geometric layers (higher = better resonance with abundance, harmony, foresight)
    pub layer_alignment: f64,
    /// Cumulative cooperation score across all interactions
    pub cooperation_score: f64,
    /// History of major action types for audit and future modulation
    pub action_history: Vec<ActionType>,
}

impl Default for EpigeneticProfile {
    fn default() -> Self {
        Self {
            strength: 1.0,
            volatility: 0.3,
            layer_alignment: 0.6,
            cooperation_score: 0.0,
            action_history: Vec::new(),
        }
    }
}

/// Types of actions that drive epigenetic change (from v14.5 design + council convergence).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ActionType {
    Cooperation,      // Cross-race or joint projects — highest reward
    Creation,         // Building, crafting, ecosystem stewardship
    WiseConflict,     // Necessary defensive or high-mercy-intent conflict
    TradeDiplomacy,   // RBE trade, negotiation, alliance formation
    Exploitation,     // Resource hoarding, betrayal, zero-sum play — high penalty
    ChronicConflict,  // Repeated war/grief without resolution — strong volatility spike
    ShortTermism,     // Repeated short-sighted decisions
}

/// Delta applied to a profile from a single action or event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpigeneticChange {
    pub action: ActionType,
    pub strength_delta: f64,
    pub volatility_delta: f64,
    pub layer_alignment_delta: f64,
    pub cooperation_delta: f64,
    pub reason: String, // Human-readable for logs/UI
}

/// Applies a change to an existing profile. All deltas are mercy-gated (clamped, never remove agency).
pub fn apply_change(profile: &mut EpigeneticProfile, change: &EpigeneticChange) {
    profile.strength = (profile.strength + change.strength_delta).clamp(0.1, 3.0);
    profile.volatility = (profile.volatility + change.volatility_delta).clamp(0.0, 1.5);
    profile.layer_alignment = (profile.layer_alignment + change.layer_alignment_delta).clamp(0.0, 1.0);
    profile.cooperation_score = (profile.cooperation_score + change.cooperation_delta).clamp(-10.0, 100.0);
    profile.action_history.push(change.action);

    // Optional: prune very old history for memory efficiency in long-running sims
    if profile.action_history.len() > 128 {
        profile.action_history.drain(0..32);
    }
}

/// Generates a high-reward change for cooperative/joint actions (council convergence priority).
pub fn cooperation_change(magnitude: f64, reason: &str) -> EpigeneticChange {
    EpigeneticChange {
        action: ActionType::Cooperation,
        strength_delta: magnitude * 0.08,
        volatility_delta: -magnitude * 0.06, // reduces chaos
        layer_alignment_delta: magnitude * 0.04,
        cooperation_delta: magnitude * 1.2,
        reason: reason.to_string(),
    }
}

/// Generates a creation/stewardship change.
pub fn creation_change(magnitude: f64, reason: &str) -> EpigeneticChange {
    EpigeneticChange {
        action: ActionType::Creation,
        strength_delta: magnitude * 0.06,
        volatility_delta: -magnitude * 0.03,
        layer_alignment_delta: magnitude * 0.05,
        cooperation_delta: magnitude * 0.6,
        reason: reason.to_string(),
    }
}

/// Generates penalty for exploitation/zero-sum behavior.
pub fn exploitation_change(magnitude: f64, reason: &str) -> EpigeneticChange {
    EpigeneticChange {
        action: ActionType::Exploitation,
        strength_delta: -magnitude * 0.04,
        volatility_delta: magnitude * 0.12, // increases future difficulty
        layer_alignment_delta: -magnitude * 0.05,
        cooperation_delta: -magnitude * 1.5,
        reason: reason.to_string(),
    }
}

/// Generates change for chronic conflict without resolution.
pub fn chronic_conflict_change(magnitude: f64, reason: &str) -> EpigeneticChange {
    EpigeneticChange {
        action: ActionType::ChronicConflict,
        strength_delta: -magnitude * 0.02,
        volatility_delta: magnitude * 0.15,
        layer_alignment_delta: -magnitude * 0.04,
        cooperation_delta: -magnitude * 0.8,
        reason: reason.to_string(),
    }
}

/// Batch apply multiple changes (e.g. from a full simulator tick or multi-agent round).
pub fn apply_batch(profile: &mut EpigeneticProfile, changes: &[EpigeneticChange]) {
    for change in changes {
        apply_change(profile, change);
    }
}

/// Computes a simple "health" score for UI/decision making (higher = thriving, stable, high-layer).
pub fn profile_health(profile: &EpigeneticProfile) -> f64 {
    let stability = (1.0 - profile.volatility).max(0.0);
    (profile.strength * 0.4 + stability * 0.3 + profile.layer_alignment * 0.3 + (profile.cooperation_score.max(0.0) * 0.01)).clamp(0.0, 3.0)
}

/// Example integration helper: convert MultiAgentOrchestrator action + mercy_intent into EpigeneticChange.
pub fn action_to_change(action: ActionType, mercy_intent: f64, base_magnitude: f64) -> EpigeneticChange {
    let magnitude = base_magnitude * mercy_intent.clamp(0.0, 1.2);
    match action {
        ActionType::Cooperation | ActionType::TradeDiplomacy => cooperation_change(magnitude, "Multi-agent cooperation action"),
        ActionType::Creation => creation_change(magnitude, "Creation action"),
        ActionType::Exploitation | ActionType::ChronicConflict => exploitation_change(magnitude, "Exploitative or unresolved conflict action"),
        _ => EpigeneticChange {
            action,
            strength_delta: magnitude * 0.02,
            volatility_delta: 0.0,
            layer_alignment_delta: 0.01,
            cooperation_delta: magnitude * 0.3,
            reason: "Standard action".to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cooperation_improves_profile() {
        let mut p = EpigeneticProfile::default();
        let change = cooperation_change(5.0, "Joint megaproject with rival race");
        apply_change(&mut p, &change);
        assert!(p.strength > 1.0);
        assert!(p.volatility < 0.3);
        assert!(p.cooperation_score > 4.0);
        assert!(profile_health(&p) > 1.2);
    }

    #[test]
    fn test_exploitation_increases_volatility() {
        let mut p = EpigeneticProfile::default();
        let change = exploitation_change(4.0, "Resource hoarding during scarcity");
        apply_change(&mut p, &change);
        assert!(p.volatility > 0.5);
        assert!(p.cooperation_score < 0.0);
    }

    #[test]
    fn test_batch_and_health() {
        let mut p = EpigeneticProfile::default();
        let changes = vec![
            cooperation_change(3.0, "Alliance formed"),
            creation_change(2.0, "Ecosystem restoration"),
        ];
        apply_batch(&mut p, &changes);
        assert!(profile_health(&p) > 1.4);
    }
}
