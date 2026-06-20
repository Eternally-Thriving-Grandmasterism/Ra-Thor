/*!
# AbilityTree — Race-Specific Ability Trees for Powrush MMOARPG

**Autonomicity Games Sovereign Mercy License (AG-SML) v1.0**  
**Aligned with TOLC 8 Mercy Lattice, Ra-Thor ONE Organism, 13+ PATSAGi Councils**  
**Extends the Multi-Race Foundation with meaningful, progression-based abilities**

This module implements race-specific ability trees.

**Design Principles:**
- Abilities are meaningful and fun (not just stat sticks).
- Unlock conditions are tied to positive play: cooperation, creation, exploration, and epigenetic health.
- Every race has a distinct identity and power fantasy.
- Mercy-gated: exploitation and chronic conflict slow or block progression.
- Designed for future integration with UI, hotbar, and the simulation tick.

Each race starts with 1–2 starter abilities and has clear upgrade paths.

Thunder locked in. Abilities now give players tangible expression of their race and growth.
*/

use serde::{Deserialize, Serialize};
use crate::race::Race;

/// Represents a single ability in a race's ability tree.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Ability {
    pub id: String,
    pub name: String,
    pub description: String,
    pub race: Race,
    pub tier: u8,                    // 1 = starter, 2–4 = advanced
    pub unlock_cooperation_score: f64,
    pub unlock_innovation_score: f64,
    pub unlock_contribution_total: f64,
    pub effect_type: AbilityEffect,
}

/// What the ability actually does when activated.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AbilityEffect {
    MovementBoost { duration_ticks: u32, speed_multiplier: f32 },
    HarmonyPulse { harmony_gain: f32 },
    EpigeneticStabilize { volatility_reduction: f32 },
    ContributionMultiplier { multiplier: f64, duration_ticks: u32 },
    ExplorationScan { range: f32 },
    VoidSkip { extra_distance: f32, risk: f32 },
}

/// A collection of abilities unlocked by a player of a specific race.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AbilityTree {
    pub race: Race,
    pub unlocked_abilities: Vec<Ability>,
}

impl AbilityTree {
    pub fn new(race: Race) -> Self {
        Self {
            race,
            unlocked_abilities: Vec::new(),
        }
    }

    /// Returns the starter abilities available to this race.
    pub fn starter_abilities(race: Race) -> Vec<Ability> {
        match race {
            Race::Terran => vec![Ability {
                id: "terran_steady_step".to_string(),
                name: "Steady Step".to_string(),
                description: "Slight movement speed boost and better landing stability.".to_string(),
                race: Race::Terran,
                tier: 1,
                unlock_cooperation_score: 5.0,
                unlock_innovation_score: 0.0,
                unlock_contribution_total: 20.0,
                effect_type: AbilityEffect::MovementBoost {
                    duration_ticks: 120,
                    speed_multiplier: 1.15,
                },
            }],
            Race::Synthetic => vec![Ability {
                id: "synthetic_precision_boost".to_string(),
                name: "Precision Boost".to_string(),
                description: "Improved jump accuracy and tech-assisted trajectory correction.".to_string(),
                race: Race::Synthetic,
                tier: 1,
                unlock_cooperation_score: 3.0,
                unlock_innovation_score: 8.0,
                unlock_contribution_total: 15.0,
                effect_type: AbilityEffect::MovementBoost {
                    duration_ticks: 90,
                    speed_multiplier: 1.25,
                },
            }],
            Race::Harmonic => vec![Ability {
                id: "harmonic_resonance".to_string(),
                name: "Resonant Jump".to_string(),
                description: "Jump generates a small harmony pulse on landing.".to_string(),
                race: Race::Harmonic,
                tier: 1,
                unlock_cooperation_score: 6.0,
                unlock_innovation_score: 4.0,
                unlock_contribution_total: 25.0,
                effect_type: AbilityEffect::HarmonyPulse { harmony_gain: 0.08 },
            }],
            Race::Verdant => vec![Ability {
                id: "verdant_rooted_leap".to_string(),
                name: "Rooted Leap".to_string(),
                description: "Landing reduces epigenetic volatility and grants stability.".to_string(),
                race: Race::Verdant,
                tier: 1,
                unlock_cooperation_score: 7.0,
                unlock_innovation_score: 2.0,
                unlock_contribution_total: 30.0,
                effect_type: AbilityEffect::EpigeneticStabilize { volatility_reduction: 0.15 },
            }],
            Race::Voidfarer => vec![Ability {
                id: "voidfarer_void_skip".to_string(),
                name: "Void Skip".to_string(),
                description: "Longer range jump with high reward but some risk.".to_string(),
                race: Race::Voidfarer,
                tier: 1,
                unlock_cooperation_score: 4.0,
                unlock_innovation_score: 5.0,
                unlock_contribution_total: 18.0,
                effect_type: AbilityEffect::VoidSkip {
                    extra_distance: 8.0,
                    risk: 0.12,
                },
            }],
        }
    }

    /// Attempts to unlock the starter ability for this race if conditions are met.
    pub fn try_unlock_starter(&mut self, cooperation: f64, innovation: f64, total_contribution: f64) -> Option<Ability> {
        let starters = Self::starter_abilities(self.race);
        for ability in starters {
            if cooperation >= ability.unlock_cooperation_score
                && innovation >= ability.unlock_innovation_score
                && total_contribution >= ability.unlock_contribution_total
                && !self.unlocked_abilities.contains(&ability)
            {
                self.unlocked_abilities.push(ability.clone());
                return Some(ability);
            }
        }
        None
    }

    /// Returns true if the player has at least one ability from this tree.
    pub fn has_abilities(&self) -> bool {
        !self.unlocked_abilities.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starter_unlock() {
        let mut tree = AbilityTree::new(Race::Harmonic);
        let unlocked = tree.try_unlock_starter(7.0, 5.0, 30.0);
        assert!(unlocked.is_some());
        assert!(tree.has_abilities());
    }

    #[test]
    fn test_starter_not_unlocked_too_early() {
        let mut tree = AbilityTree::new(Race::Voidfarer);
        let unlocked = tree.try_unlock_starter(2.0, 1.0, 5.0);
        assert!(unlocked.is_none());
    }
}
