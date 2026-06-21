/*!
# AbilityTree — Race-Specific Ability Trees for Powrush MMOARPG

**Autonomicity Games Sovereign Mercy License (AG-SML) v1.0**  
**Aligned with TOLC 8 Mercy Lattice, Ra-Thor ONE Organism, 13+ PATSAGi Councils**  
**Mutation-Specific Synergy Chains with Stage Progression (v1.9)**

This version adds **chain stage progression mechanics** — chains now evolve over time based on sustained conditions (high harmony, consistent contribution, epigenetic stability). Stages unlock escalating bonuses, creating true long-term evolutionary mastery paths.
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
    pub requires_ability: Option<String>,
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

/// Represents an active synergy bonus from ability combinations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynergyBonus {
    pub name: String,
    pub description: String,
    pub bonus_type: SynergyType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynergyType {
    HarmonyAmplification { multiplier: f32 },
    ContributionBoost { multiplier: f64 },
    MovementEfficiency { multiplier: f32 },
    EpigeneticResilience { reduction: f32 },
    GlobalCooldownReduction { percent: f32 },
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
    /// Tracks progression points for each mutation-specific synergy chain.
    /// Higher values unlock stronger stage bonuses.
    chain_progress: HashMap<String, u32>,
}

impl AbilityTree {
    pub fn new(race: Race) -> Self {
        Self {
            race,
            unlocked_abilities: Vec::new(),
            cooldowns: HashMap::new(),
            chain_progress: HashMap::new(),
        }
    }

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

    fn has_prerequisite(&self, ability: &Ability) -> bool {
        match &ability.requires_ability {
            Some(req_id) => self.unlocked_abilities.iter().any(|a| a.id == *req_id),
            None => true,
        }
    }

    pub fn calculate_synergies(&self) -> Vec<SynergyBonus> {
        let mut bonuses = Vec::new();
        let unlocked_ids: Vec<&String> = self.unlocked_abilities.iter().map(|a| &a.id).collect();

        match self.race {
            Race::Terran => {
                if unlocked_ids.contains(&&"terran_steady_step".to_string())
                    && unlocked_ids.contains(&&"terran_community_bond".to_string())
                {
                    bonuses.push(SynergyBonus {
                        name: "Terran Unity".to_string(),
                        description: "+15% Harmony gain from all sources when you have both Steady Step and Community Bond.".to_string(),
                        bonus_type: SynergyType::HarmonyAmplification { multiplier: 1.15 },
                    });
                }
                if unlocked_ids.contains(&&"terran_fortress_stand".to_string()) {
                    bonuses.push(SynergyBonus {
                        name: "Unbreakable Line".to_string(),
                        description: "+25% Contribution multiplier while defending when Fortress Stand is unlocked.".to_string(),
                        bonus_type: SynergyType::ContributionBoost { multiplier: 1.25 },
                    });
                }
            }
            Race::Harmonic => {
                if unlocked_ids.contains(&&"harmonic_resonance".to_string())
                    && unlocked_ids.contains(&&"harmonic_resonant_field".to_string())
                {
                    bonuses.push(SynergyBonus {
                        name: "Harmonic Resonance Cascade".to_string(),
                        description: "+20% Harmony pulse strength when both Resonant Jump and Resonant Field are active.".to_string(),
                        bonus_type: SynergyType::HarmonyAmplification { multiplier: 1.20 },
                    });
                }
            }
            Race::Synthetic => {
                if unlocked_ids.contains(&&"synthetic_overclock".to_string())
                    && unlocked_ids.contains(&&"synthetic_systems_mastery".to_string())
                {
                    bonuses.push(SynergyBonus {
                        name: "Synthetic Efficiency".to_string(),
                        description: "-10% cooldown on all abilities when both Overclock and Systems Mastery are unlocked.".to_string(),
                        bonus_type: SynergyType::GlobalCooldownReduction { percent: 0.10 },
                    });
                }
            }
            Race::Verdant => {
                if unlocked_ids.contains(&&"verdant_lifebloom".to_string())
                    && unlocked_ids.contains(&&"verdant_ancient_growth".to_string())
                {
                    bonuses.push(SynergyBonus {
                        name: "Verdant Legacy".to_string(),
                        description: "+30% Epigenetic stability from all sources when both Lifebloom and Ancient Growth are unlocked.".to_string(),
                        bonus_type: SynergyType::EpigeneticResilience { reduction: 0.30 },
                    });
                }
            }
            Race::Voidfarer => {
                if unlocked_ids.contains(&&"voidfarer_phase_shift".to_string())
                    && unlocked_ids.contains(&&"voidfarer_singularity_drive".to_string())
                {
                    bonuses.push(SynergyBonus {
                        name: "Voidfarer Singularity".to_string(),
                        description: "+15% Movement efficiency and reduced risk on long-range abilities.".to_string(),
                        bonus_type: SynergyType::MovementEfficiency { multiplier: 1.15 },
                    });
                }
            }
        }

        bonuses
    }

    /// Returns current stage (0-2) for a given chain key.
    pub fn get_chain_stage(&self, chain_key: &str) -> u8 {
        let progress = self.chain_progress.get(chain_key).copied().unwrap_or(0);
        if progress >= 70 { 2 } else if progress >= 30 { 1 } else { 0 }
    }

    /// Advances chain progression based on sustained positive conditions.
    /// Called from simulator tick when the chain is active.
    pub fn progress_chain_stages(&mut self, chain_key: &str, harmony_level: f32, recent_contribution: f64, volatility: f32) {
        let progress = self.chain_progress.entry(chain_key.to_string()).or_insert(0);

        let mut advance = 0u32;

        if harmony_level > 1.8 {
            advance += 2;
        } else if harmony_level > 1.2 {
            advance += 1;
        }

        if recent_contribution > 8.0 {
            advance += 1;
        }

        if volatility < 0.6 {
            advance += 1;
        }

        if advance > 0 {
            *progress = (*progress + advance).min(120);
        }

        // Slow natural decay if conditions are poor
        if harmony_level < 0.9 && volatility > 1.0 {
            if *progress > 0 {
                *progress -= 1;
            }
        }
    }

    /// Mutation-Specific Synergy Chains with Stage Progression (v1.9)
    pub fn calculate_mutation_synergy_chains(&self, active_mutations: &[String]) -> Vec<SynergyBonus> {
        let mut bonuses = Vec::new();
        let unlocked_ids: Vec<&String> = self.unlocked_abilities.iter().map(|a| &a.id).collect();

        for mutation in active_mutations {
            match mutation.as_str() {
                "harmonic_rebirth" => {
                    if (self.race == Race::Harmonic || self.race == Race::Verdant)
                        && unlocked_ids.contains(&&"harmonic_resonant_field".to_string())
                        && unlocked_ids.contains(&&"harmonic_cosmic_attunement".to_string())
                    {
                        let stage = self.get_chain_stage("redemption_cascade");

                        bonuses.push(SynergyBonus {
                            name: format!("Redemption Cascade Chain (Stage {})", stage),
                            description: match stage {
                                0 => "Harmonic Rebirth + full Harmonic tree: Foundational harmony repair and corruption resistance.".to_string(),
                                1 => "Stage 1: Amplified repair, stronger passive positive drift, reduced backlash severity.".to_string(),
                                _ => "Stage 2 (Mastered): Maximum redemptive power — powerful ongoing epigenetic healing and harmony mastery.".to_string(),
                            },
                            bonus_type: SynergyType::HarmonyAmplification { multiplier: 1.25 + (stage as f32 * 0.12) },
                        });

                        if stage >= 1 {
                            bonuses.push(SynergyBonus {
                                name: "Redemption Cascade (Resilience)".to_string(),
                                description: "Stage-enhanced epigenetic resilience and repair speed.".to_string(),
                                bonus_type: SynergyType::EpigeneticResilience { reduction: 0.20 + (stage as f32 * 0.12) },
                            });
                        }

                        if stage >= 2 {
                            bonuses.push(SynergyBonus {
                                name: "Redemption Cascade (Evolved)".to_string(),
                                description: "Mastered stage: Chance for further positive epigenetic evolution and permanent stability gains.".to_string(),
                                bonus_type: SynergyType::EpigeneticResilience { reduction: 0.45 },
                            });
                        }
                    }
                }
                "volatile_surge" => {
                    if unlocked_ids.contains(&&"synthetic_overclock".to_string())
                        && unlocked_ids.contains(&&"synthetic_systems_mastery".to_string())
                    {
                        let stage = self.get_chain_stage("surge_overclock");
                        bonuses.push(SynergyBonus {
                            name: format!("Surge Overclock Chain (Stage {})", stage),
                            description: match stage {
                                0 => "Volatile Surge + Synthetic power: Amplified contribution at risk of backlash.".to_string(),
                                1 => "Stage 1: Greater power spikes, controlled volatility channeling.".to_string(),
                                _ => "Stage 2 (Mastered): High-risk mastery — massive temporary power with managed backlash.".to_string(),
                            },
                            bonus_type: SynergyType::ContributionBoost { multiplier: 1.35 + (stage as f64 * 0.18) },
                        });
                    }
                }
                "corrupted_echo" => {
                    if unlocked_ids.contains(&&"voidfarer_phase_shift".to_string())
                        && unlocked_ids.contains(&&"voidfarer_singularity_drive".to_string())
                    {
                        let stage = self.get_chain_stage("corrupted_singularity");
                        bonuses.push(SynergyBonus {
                            name: format!("Corrupted Singularity Chain (Stage {})", stage),
                            description: match stage {
                                0 => "Corrupted Echo + Voidfarer high-risk path: High gains with corruption cost.".to_string(),
                                1 => "Stage 1: Dangerous power, partial corruption mitigation through mastery.".to_string(),
                                _ => "Stage 2 (Mastered): Corrupted power fully weaponized — extreme contribution with slow corruption.".to_string(),
                            },
                            bonus_type: SynergyType::ContributionBoost { multiplier: 1.50 + (stage as f64 * 0.20) },
                        });
                    }
                }
                _ => {}
            }
        }

        bonuses
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synergy_calculation() {
        let mut tree = AbilityTree::new(Race::Terran);
        let _ = tree.try_unlock_starter(30.0, 15.0, 150.0);
        let synergies = tree.calculate_synergies();
        assert!(!synergies.is_empty());
    }

    #[test]
    fn test_mutation_synergy_chain_with_stages() {
        let mut tree = AbilityTree::new(Race::Harmonic);
        let _ = tree.try_unlock_starter(60.0, 40.0, 350.0);
        tree.progress_chain_stages("redemption_cascade", 2.0, 12.0, 0.4);
        tree.progress_chain_stages("redemption_cascade", 2.1, 15.0, 0.3);
        let chains = tree.calculate_mutation_synergy_chains(&vec!["harmonic_rebirth".to_string()]);
        assert!(!chains.is_empty());
        let stage = tree.get_chain_stage("redemption_cascade");
        assert!(stage >= 0);
    }
}
