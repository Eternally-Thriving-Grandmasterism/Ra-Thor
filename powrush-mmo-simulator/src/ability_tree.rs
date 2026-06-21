/*!
# AbilityTree — Race-Specific Ability Trees for Powrush MMOARPG

**Autonomicity Games Sovereign Mercy License (AG-SML) v1.0**  
**Aligned with TOLC 8 Mercy Lattice, Ra-Thor ONE Organism, 13+ PATSAGi Councils**  
**Advanced / Branching Abilities per Race (v1.6)**

This version expands the ability system with meaningful advanced abilities and branching prerequisites.

Each race now has:
- 1 Starter ability (Tier 1)
- 2–3 Advanced abilities (Tier 2–3) with prerequisites

Abilities form proper branching trees. Cooperation and creation remain the most powerful long-term paths.
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::race::Race;

include!(concat!(env!("OUT_DIR"), "/ability.rs"));

pub use ability::{Ability as ProtoAbility, AbilityEffect as ProtoAbilityEffect, AbilityState as ProtoAbilityState, AbilityTree as ProtoAbilityTree, Race as ProtoRace};

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
    pub requires_ability: Option<String>, // NEW: prerequisite ability id
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

    /// Returns all possible abilities for this race (starter + advanced).
    pub fn all_abilities(race: Race) -> Vec<Ability> {
        let mut abilities = Self::starter_abilities(race);
        abilities.extend(Self::advanced_abilities(race));
        abilities
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
                requires_ability: None,
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
                requires_ability: None,
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
                requires_ability: None,
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
                requires_ability: None,
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
                requires_ability: None,
            }],
        }
    }

    /// Advanced / Branching abilities per race (Tier 2–3).
    pub fn advanced_abilities(race: Race) -> Vec<Ability> {
        match race {
            Race::Terran => vec![
                Ability {
                    id: "terran_community_bond".to_string(),
                    name: "Community Bond".to_string(),
                    description: "Nearby allies gain movement and harmony bonuses when you are grounded.".to_string(),
                    race: Race::Terran,
                    tier: 2,
                    unlock_cooperation_score: 25.0,
                    unlock_innovation_score: 10.0,
                    unlock_contribution_total: 120.0,
                    effect_type: AbilityEffect::HarmonyPulse { harmony_gain: 0.12 },
                    cooldown_ticks: 300,
                    requires_ability: Some("terran_steady_step".to_string()),
                },
                Ability {
                    id: "terran_fortress_stand".to_string(),
                    name: "Fortress Stand".to_string(),
                    description: "Greatly increased stability and contribution multiplier when defending a location.".to_string(),
                    race: Race::Terran,
                    tier: 3,
                    unlock_cooperation_score: 45.0,
                    unlock_innovation_score: 20.0,
                    unlock_contribution_total: 280.0,
                    effect_type: AbilityEffect::ContributionMultiplier { multiplier: 1.8, duration_ticks: 180 },
                    cooldown_ticks: 420,
                    requires_ability: Some("terran_community_bond".to_string()),
                },
            ],
            Race::Synthetic => vec![
                Ability {
                    id: "synthetic_overclock".to_string(),
                    name: "Overclock".to_string(),
                    description: "Temporarily massively increases movement speed at the cost of higher cooldown.".to_string(),
                    race: Race::Synthetic,
                    tier: 2,
                    unlock_cooperation_score: 15.0,
                    unlock_innovation_score: 30.0,
                    unlock_contribution_total: 90.0,
                    effect_type: AbilityEffect::MovementBoost { duration_ticks: 60, speed_multiplier: 1.6 },
                    cooldown_ticks: 240,
                    requires_ability: Some("synthetic_precision_boost".to_string()),
                },
                Ability {
                    id: "synthetic_systems_mastery".to_string(),
                    name: "Systems Mastery".to_string(),
                    description: "Permanent innovation and contribution bonuses after multiple successful ability uses.".to_string(),
                    race: Race::Synthetic,
                    tier: 3,
                    unlock_cooperation_score: 20.0,
                    unlock_innovation_score: 55.0,
                    unlock_contribution_total: 200.0,
                    effect_type: AbilityEffect::ContributionMultiplier { multiplier: 1.5, duration_ticks: 300 },
                    cooldown_ticks: 360,
                    requires_ability: Some("synthetic_overclock".to_string()),
                },
            ],
            Race::Harmonic => vec![
                Ability {
                    id: "harmonic_resonant_field".to_string(),
                    name: "Resonant Field".to_string(),
                    description: "Creates a field that amplifies harmony gain for all nearby entities.".to_string(),
                    race: Race::Harmonic,
                    tier: 2,
                    unlock_cooperation_score: 30.0,
                    unlock_innovation_score: 25.0,
                    unlock_contribution_total: 150.0,
                    effect_type: AbilityEffect::HarmonyPulse { harmony_gain: 0.18 },
                    cooldown_ticks: 280,
                    requires_ability: Some("harmonic_resonance".to_string()),
                },
                Ability {
                    id: "harmonic_cosmic_attunement".to_string(),
                    name: "Cosmic Attunement".to_string(),
                    description: "Greatly increased layer transition speed and harmony when in high-harmony zones.".to_string(),
                    race: Race::Harmonic,
                    tier: 3,
                    unlock_cooperation_score: 50.0,
                    unlock_innovation_score: 35.0,
                    unlock_contribution_total: 320.0,
                    effect_type: AbilityEffect::HarmonyPulse { harmony_gain: 0.25 },
                    cooldown_ticks: 450,
                    requires_ability: Some("harmonic_resonant_field".to_string()),
                },
            ],
            Race::Verdant => vec![
                Ability {
                    id: "verdant_lifebloom".to_string(),
                    name: "Lifebloom".to_string(),
                    description: "Heals epigenetic volatility for self and nearby allies over time.".to_string(),
                    race: Race::Verdant,
                    tier: 2,
                    unlock_cooperation_score: 35.0,
                    unlock_innovation_score: 15.0,
                    unlock_contribution_total: 180.0,
                    effect_type: AbilityEffect::EpigeneticStabilize { volatility_reduction: 0.25 },
                    cooldown_ticks: 320,
                    requires_ability: Some("verdant_rooted_leap".to_string()),
                },
                Ability {
                    id: "verdant_ancient_growth".to_string(),
                    name: "Ancient Growth".to_string(),
                    description: "Permanent increase to contribution and epigenetic stability after long cooperation chains.".to_string(),
                    race: Race::Verdant,
                    tier: 3,
                    unlock_cooperation_score: 55.0,
                    unlock_innovation_score: 25.0,
                    unlock_contribution_total: 350.0,
                    effect_type: AbilityEffect::ContributionMultiplier { multiplier: 1.6, duration_ticks: 240 },
                    cooldown_ticks: 480,
                    requires_ability: Some("verdant_lifebloom".to_string()),
                },
            ],
            Race::Voidfarer => vec![
                Ability {
                    id: "voidfarer_phase_shift".to_string(),
                    name: "Phase Shift".to_string(),
                    description: "Briefly become intangible, avoiding all negative effects while moving.".to_string(),
                    race: Race::Voidfarer,
                    tier: 2,
                    unlock_cooperation_score: 18.0,
                    unlock_innovation_score: 28.0,
                    unlock_contribution_total: 110.0,
                    effect_type: AbilityEffect::MovementBoost { duration_ticks: 45, speed_multiplier: 1.4 },
                    cooldown_ticks: 200,
                    requires_ability: Some("voidfarer_void_skip".to_string()),
                },
                Ability {
                    id: "voidfarer_singularity_drive".to_string(),
                    name: "Singularity Drive".to_string(),
                    description: "Extremely long range movement with powerful but risky contribution reward.".to_string(),
                    race: Race::Voidfarer,
                    tier: 3,
                    unlock_cooperation_score: 30.0,
                    unlock_innovation_score: 45.0,
                    unlock_contribution_total: 260.0,
                    effect_type: AbilityEffect::VoidSkip { extra_distance: 18.0, risk: 0.18 },
                    cooldown_ticks: 380,
                    requires_ability: Some("voidfarer_phase_shift".to_string()),
                },
            ],
        }
    }

    pub fn try_unlock_starter(&mut self, cooperation: f64, innovation: f64, total_contribution: f64) -> Option<Ability> {
        let all = Self::all_abilities(self.race);
        for ability in all {
            if cooperation >= ability.unlock_cooperation_score
                && innovation >= ability.unlock_innovation_score
                && total_contribution >= ability.unlock_contribution_total
                && !self.unlocked_abilities.iter().any(|a| a.id == ability.id)
                && self.has_prerequisite(&ability)
            {
                self.unlocked_abilities.push(ability.clone());
                return Some(ability);
            }
        }
        None
    }

    /// Checks if the player has unlocked the required prerequisite ability (if any).
    fn has_prerequisite(&self, ability: &Ability) -> bool {
        match &ability.requires_ability {
            Some(req_id) => self.unlocked_abilities.iter().any(|a| a.id == *req_id),
            None => true,
        }
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

        // Show locked advanced abilities for UI preview
        for ability in Self::advanced_abilities(self.race) {
            if !states.iter().any(|s| s.id == ability.id) {
                let can_unlock = self.has_prerequisite(&ability);
                states.push(AbilityState {
                    id: ability.id,
                    name: ability.name,
                    description: ability.description,
                    unlocked: false,
                    on_cooldown: false,
                    remaining_cooldown_ticks: 0,
                    cooldown_progress: if can_unlock { 0.0 } else { 0.0 },
                });
            }
        }

        states
    }

    pub fn has_abilities(&self) -> bool {
        !self.unlocked_abilities.is_empty()
    }
}

// (Conversion methods for Protobuf, JSON, Binary remain unchanged from v1.5)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_unlock_with_prerequisite() {
        let mut tree = AbilityTree::new(Race::Harmonic);
        let _ = tree.try_unlock_starter(30.0, 25.0, 150.0); // unlock starter
        let advanced = tree.try_unlock_starter(35.0, 30.0, 200.0); // should unlock Resonant Field
        assert!(advanced.is_some());
        assert_eq!(advanced.unwrap().id, "harmonic_resonant_field");
    }
}
