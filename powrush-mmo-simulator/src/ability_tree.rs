/*!
# AbilityTree — Race-Specific Ability Trees for Powrush MMOARPG

**Autonomicity Games Sovereign Mercy License (AG-SML) v1.0**  
**Aligned with TOLC 8 Mercy Lattice, Ra-Thor ONE Organism, 13+ PATSAGi Councils**  
**Extends the Multi-Race Foundation with meaningful, progression-based abilities + Cooldown Mechanics**

This module implements race-specific ability trees with cooldown support.

**Cooldown Design:**
- Each ability has a base cooldown (in ticks).
- `try_use_ability` checks if enough ticks have passed since last use.
- Cooldowns are per-ability and reset on successful use.
- Designed for future hotbar / player-controlled activation.

Thunder locked in. Abilities now have strategic timing.
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::race::Race;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Ability {
    pub id: String,
    pub name: String,
    pub description: String,
    pub race: Race,
    pub tier: u8,
    pub unlock_cooperation_score: f64,
    pub unlock_innovation_score: f64,
    pub unlock_contribution_total: f64,
    pub effect_type: AbilityEffect,
    pub cooldown_ticks: u32, // NEW: base cooldown
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AbilityEffect {
    MovementBoost { duration_ticks: u32, speed_multiplier: f32 },
    HarmonyPulse { harmony_gain: f32 },
    EpigeneticStabilize { volatility_reduction: f32 },
    ContributionMultiplier { multiplier: f64, duration_ticks: u32 },
    ExplorationScan { range: f32 },
    VoidSkip { extra_distance: f32, risk: f32 },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AbilityTree {
    pub race: Race,
    pub unlocked_abilities: Vec<Ability>,
    cooldowns: HashMap<String, u64>, // ability_id -> last used tick
}

impl AbilityTree {
    pub fn new(race: Race) -> Self {
        Self {
            race,
            unlocked_abilities: Vec::new(),
            cooldowns: HashMap::new(),
        }
    }

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
                effect_type: AbilityEffect::MovementBoost { duration_ticks: 120, speed_multiplier: 1.15 },
                cooldown_ticks: 180,
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
                effect_type: AbilityEffect::MovementBoost { duration_ticks: 90, speed_multiplier: 1.25 },
                cooldown_ticks: 150,
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
                cooldown_ticks: 200,
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
                cooldown_ticks: 220,
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
                effect_type: AbilityEffect::VoidSkip { extra_distance: 8.0, risk: 0.12 },
                cooldown_ticks: 160,
            }],
        }
    }

    pub fn try_unlock_starter(&mut self, cooperation: f64, innovation: f64, total_contribution: f64) -> Option<Ability> {
        let starters = Self::starter_abilities(self.race);
        for ability in starters {
            if cooperation >= ability.unlock_cooperation_score
                && innovation >= ability.unlock_innovation_score
                && total_contribution >= ability.unlock_contribution_total
                && !self.unlocked_abilities.iter().any(|a| a.id == ability.id)
            {
                self.unlocked_abilities.push(ability.clone());
                return Some(ability);
            }
        }
        None
    }

    /// Returns true if the ability is currently on cooldown.
    pub fn is_on_cooldown(&self, ability_id: &str, current_tick: u64) -> bool {
        if let Some(&last_used) = self.cooldowns.get(ability_id) {
            if let Some(ability) = self.unlocked_abilities.iter().find(|a| a.id == ability_id) {
                return current_tick < last_used + ability.cooldown_ticks as u64;
            }
        }
        false
    }

    /// Attempts to use an ability. Returns true if successful (off cooldown).
    /// Updates the cooldown timestamp on success.
    pub fn try_use_ability(&mut self, ability_id: &str, current_tick: u64) -> bool {
        if self.is_on_cooldown(ability_id, current_tick) {
            return false;
        }

        if self.unlocked_abilities.iter().any(|a| a.id == ability_id) {
            self.cooldowns.insert(ability_id.to_string(), current_tick);
            return true;
        }

        false
    }

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
    }

    #[test]
    fn test_cooldown_mechanics() {
        let mut tree = AbilityTree::new(Race::Harmonic);
        let _ = tree.try_unlock_starter(7.0, 5.0, 30.0);

        assert!(tree.try_use_ability("harmonic_resonance", 100));
        assert!(tree.is_on_cooldown("harmonic_resonance", 150));
        assert!(!tree.try_use_ability("harmonic_resonance", 150));
        assert!(tree.try_use_ability("harmonic_resonance", 350));
    }
}
