/*!
# PlayerContribution — RBE Player-Level Contribution Tracking for Powrush MMOARPG

**Autonomicity Games Sovereign Mercy License (AG-SML) v1.0**  
**Aligned with TOLC 8 Mercy Lattice, Ra-Thor ONE Organism, 13+ PATSAGi Councils**  
**Implements Council Convergence Priority: Visible Economic Feedback + Player Impact on RBE**

This module provides production-grade tracking of individual player contributions to the Resource Based Economy.

Key goals:
- Make cooperation and creation visibly rewarding at the player level.
- Feed into EpigeneticModulation (cooperation_score) and GeometricHarmony.
- Enable future RREL (Real Estate Lattice) and personal abundance dashboards.
- Mercy-gated: exploitation and chronic conflict reduce future contribution multipliers.

Designed for tight integration with PowrushMMOSimulator, RBEconomy, and the movement/epigenetic systems.

Thunder locked in. Player actions now have persistent, visible economic meaning.
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Types of player contributions that affect the RBE and player state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContributionType {
    Production,      // Building, crafting, resource creation
    Innovation,      // New ideas, research, system improvements
    Cooperation,     // Joint projects, helping others, diplomacy
    Exploration,     // Mapping, discovery, risk-taking for the collective
    Maintenance,     // Repair, defense, infrastructure upkeep
    Exploitation,    // Harmful extraction (mechanically penalized)
}

/// A single recorded contribution from a player.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerContribution {
    pub player_id: u64,
    pub contribution_type: ContributionType,
    pub amount: f64,
    pub timestamp_tick: u64,
    pub cooperation_multiplier: f64, // Higher when done with others
}

/// Tracks all contributions for a single player.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlayerContributionProfile {
    pub total_contributions: f64,
    pub cooperation_score: f64,
    pub innovation_score: f64,
    pub recent_contributions: Vec<PlayerContribution>,
    pub last_updated_tick: u64,
}

/// Main engine for recording and querying player contributions.
pub struct PlayerContributionTracker {
    pub profiles: HashMap<u64, PlayerContributionProfile>,
}

impl Default for PlayerContributionTracker {
    fn default() -> Self {
        Self {
            profiles: HashMap::new(),
        }
    }
}

impl PlayerContributionTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new contribution from a player.
    /// Automatically updates scores and applies mercy-gating for exploitation.
    pub fn record_contribution(
        &mut self,
        player_id: u64,
        contribution_type: ContributionType,
        amount: f64,
        current_tick: u64,
        cooperation_multiplier: f64,
    ) {
        let profile = self.profiles.entry(player_id).or_default();

        let effective_amount = if contribution_type == ContributionType::Exploitation {
            amount * 0.3 // Strong penalty
        } else {
            amount * cooperation_multiplier
        };

        let contribution = PlayerContribution {
            player_id,
            contribution_type,
            amount: effective_amount,
            timestamp_tick: current_tick,
            cooperation_multiplier,
        };

        profile.recent_contributions.push(contribution);
        if profile.recent_contributions.len() > 50 {
            profile.recent_contributions.remove(0); // Keep last 50
        }

        profile.total_contributions += effective_amount;
        profile.last_updated_tick = current_tick;

        // Update specialized scores
        match contribution_type {
            ContributionType::Cooperation => {
                profile.cooperation_score = (profile.cooperation_score * 0.9 + effective_amount * 0.1).min(100.0);
            }
            ContributionType::Innovation => {
                profile.innovation_score = (profile.innovation_score * 0.9 + effective_amount * 0.1).min(100.0);
            }
            _ => {}
        }
    }

    /// Get a player's current cooperation score (feeds into EpigeneticModulation).
    pub fn get_cooperation_score(&self, player_id: u64) -> f64 {
        self.profiles.get(&player_id)
            .map(|p| p.cooperation_score)
            .unwrap_or(0.0)
    }

    /// Calculate how much this player's contributions should boost RBE abundance this tick.
    pub fn calculate_rbe_impact(&self, player_id: u64) -> f64 {
        if let Some(profile) = self.profiles.get(&player_id) {
            let base = (profile.total_contributions * 0.001).min(2.5);
            let cooperation_bonus = profile.cooperation_score * 0.02;
            base + cooperation_bonus
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_score() {
        let mut tracker = PlayerContributionTracker::new();
        tracker.record_contribution(1, ContributionType::Cooperation, 50.0, 100, 1.5);
        assert!(tracker.get_cooperation_score(1) > 0.0);
    }

    #[test]
    fn test_exploitation_penalty() {
        let mut tracker = PlayerContributionTracker::new();
        tracker.record_contribution(2, ContributionType::Exploitation, 100.0, 100, 1.0);
        let profile = tracker.profiles.get(&2).unwrap();
        assert!(profile.total_contributions < 100.0); // Penalized
    }
}
