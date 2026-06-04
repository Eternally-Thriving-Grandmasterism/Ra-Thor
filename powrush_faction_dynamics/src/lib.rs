/*!
# Powrush Faction Dynamics — Reputation System

Production-grade faction reputation system for Powrush RBE MMO.

## Design Principles (Thoughtful & Professional)
- Reputation is earned through actions (council success, contributions, harmony maintenance).
- Reputation provides influence multipliers on economic and political power.
- Reputation changes are transparent and event-driven.
- Integrates cleanly with RBEconomy (contribution weight) and council proposals (valence bonus).
- Mercy-aligned: High-reputation factions can champion more universal policies, but abuse leads to decay.

This crate provides the foundation for meaningful faction identity and long-term dynamics in the simulation.
*/

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Faction {
    Forge,
    Evolutionary,
    Harmony,
}

impl Faction {
    pub fn as_str(&self) -> &'static str {
        match self {
            Faction::Forge => "Forge",
            Faction::Evolutionary => "Evolutionary",
            Faction::Harmony => "Harmony",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReputationEvent {
    pub event_type: ReputationEventType,
    pub magnitude: f64, // Positive or negative impact
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReputationEventType {
    CouncilProposalSuccess,
    MajorEconomicContribution,
    HarmonyMaintenance,
    ConflictResolution,
    PolicyAdvocacy,
}

#[derive(Debug, Clone)]
pub struct FactionReputation {
    pub score: f64,           // 0.0 (pariah) to 1.0 (legendary)
    pub history: Vec<ReputationEvent>,
}

impl FactionReputation {
    pub fn new(initial_score: f64) -> Self {
        Self {
            score: initial_score.clamp(0.0, 1.0),
            history: Vec::new(),
        }
    }

    /// Apply a reputation event. Changes are clamped and logged.
    pub fn apply_event(&mut self, event: ReputationEvent) {
        let change = event.magnitude.clamp(-0.15, 0.15);
        self.score = (self.score + change).clamp(0.0, 1.0);
        self.history.push(event);

        // Keep history bounded for production use
        if self.history.len() > 50 {
            self.history.remove(0);
        }
    }

    /// Influence multiplier for contributions and proposal weight (0.7 – 1.35)
    pub fn influence_multiplier(&self) -> f64 {
        0.7 + (self.score * 0.65)
    }

    /// Bonus to council proposal valence when this faction initiates
    pub fn council_valence_bonus(&self) -> f64 {
        (self.score - 0.5) * 0.2
    }
}

/// Manages reputation for all factions in the simulation
#[derive(Debug, Clone)]
pub struct ReputationSystem {
    pub factions: HashMap<String, FactionReputation>,
}

impl ReputationSystem {
    pub fn new() -> Self {
        let mut factions = HashMap::new();
        factions.insert("Forge".to_string(), FactionReputation::new(0.75));
        factions.insert("Evolutionary".to_string(), FactionReputation::new(0.72));
        factions.insert("Harmony".to_string(), FactionReputation::new(0.88));

        Self { factions }
    }

    pub fn get(&self, faction: &str) -> Option<&FactionReputation> {
        self.factions.get(faction)
    }

    pub fn get_mut(&mut self, faction: &str) -> Option<&mut FactionReputation> {
        self.factions.get_mut(faction)
    }

    /// Apply an event to a faction (used from simulator tick)
    pub fn record_event(&mut self, faction: &str, event: ReputationEvent) {
        if let Some(rep) = self.factions.get_mut(faction) {
            rep.apply_event(event);
        }
    }

    /// Get combined influence for RBE contribution weighting
    pub fn get_contribution_multiplier(&self, faction: &str) -> f64 {
        self.get(faction)
            .map(|r| r.influence_multiplier())
            .unwrap_or(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reputation_adjustment_and_multiplier() {
        let mut system = ReputationSystem::new();
        let event = ReputationEvent {
            event_type: ReputationEventType::CouncilProposalSuccess,
            magnitude: 0.12,
            description: "Led successful harmony policy".to_string(),
        };
        system.record_event("Harmony", event);

        let rep = system.get("Harmony").unwrap();
        assert!(rep.score > 0.88);
        assert!(rep.influence_multiplier() > 1.2);
    }
}
