/*!
# AbilityTree — Race-Specific Ability Trees for Powrush MMOARPG

**Autonomicity Games Sovereign Mercy License (AG-SML) v1.0**  
**Aligned with TOLC 8 Mercy Lattice, Ra-Thor ONE Organism, 13+ PATSAGi Councils**  
**Full Ability State Serialization Support**

This module now provides complete, production-ready serialization for ability state.

**Serialization Features:**
- `Ability`, `AbilityEffect`, `AbilityState`, and `AbilityTree` are fully `Serialize` + `Deserialize`.
- Convenience methods: `to_json()` and `from_json()` for easy save/load and network use.
- Cooldown state is preserved across serialization.

Perfect for player save files, database storage, and client-server sync.

Thunder locked in. Ability state is now fully serializable.
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
    pub cooldown_ticks: u32,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbilityState {
    pub id: String,
    pub name: String,
    pub description: String,
    pub unlocked: bool,
    pub on_cooldown: bool,
    pub remaining_cooldown_ticks: u32,
    pub cooldown_progress: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AbilityTree {
    pub race: Race,
    pub unlocked_abilities: Vec<Ability>,
    cooldowns: HashMap<String, u64>,
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

    pub fn is_on_cooldown(&self, ability_id: &str, current_tick: u64) -> bool {
        if let Some(&last_used) = self.cooldowns.get(ability_id) {
            if let Some(ability) = self.unlocked_abilities.iter().find(|a| a.id == ability_id) {
                return current_tick < last_used + ability.cooldown_ticks as u64;
            }
        }
        false
    }

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

    pub fn get_ability_states(&self, current_tick: u64) -> Vec<AbilityState> {
        let mut states = Vec::new();

        for ability in &self.unlocked_abilities {
            let on_cooldown = self.is_on_cooldown(&ability.id, current_tick);
            let remaining = if on_cooldown {
                if let Some(&last_used) = self.cooldowns.get(&ability.id) {
                    (last_used + ability.cooldown_ticks as u64 - current_tick) as u32
                } else { 0 }
            } else { 0 };

            let progress = if on_cooldown && ability.cooldown_ticks > 0 {
                1.0 - (remaining as f32 / ability.cooldown_ticks as f32)
            } else { 1.0 };

            states.push(AbilityState {
                id: ability.id.clone(),
                name: ability.name.clone(),
                description: ability.description.clone(),
                unlocked: true,
                on_cooldown,
                remaining_cooldown_ticks: remaining,
                cooldown_progress: progress.clamp(0.0, 1.0),
            });
        }

        for ability in Self::starter_abilities(self.race) {
            if !states.iter().any(|s| s.id == ability.id) {
                states.push(AbilityState {
                    id: ability.id,
                    name: ability.name,
                    description: ability.description,
                    unlocked: false,
                    on_cooldown: false,
                    remaining_cooldown_ticks: 0,
                    cooldown_progress: 0.0,
                });
            }
        }

        states
    }

    /// NEW: Serialize the entire AbilityTree to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// NEW: Deserialize an AbilityTree from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn has_abilities(&self) -> bool {
        !self.unlocked_abilities.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization_roundtrip() {
        let mut tree = AbilityTree::new(Race::Harmonic);
        let _ = tree.try_unlock_starter(7.0, 5.0, 30.0);
        let json = tree.to_json().unwrap();
        let restored = AbilityTree::from_json(&json).unwrap();
        assert_eq!(tree.unlocked_abilities.len(), restored.unlocked_abilities.len());
    }
}
