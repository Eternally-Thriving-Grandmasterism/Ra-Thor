/*!
# Powrush MMOARPG Simulator — Cross-Race Diplomacy Module

**Autonomicity Games Sovereign Mercy License (AG-SML) v1.0**  
**Aligned with TOLC 8 Mercy Lattice, Ra-Thor ONE Organism, 13+ PATSAGi Councils**  
**v1.0 — Foundational Cross-Race Diplomacy Mechanics**

Enables meaningful diplomatic relations between the 5 races (Terran, Synthetic, Harmonic, Verdant, Voidfarer).
High diplomacy between a player's unlocked races creates powerful positive feedback:
- Boosts global harmony
- Accelerates cross-race synergy chain progression
- Provides epigenetic stability and repair bonuses
- Reduces backlash risk for hybrid builds

Creates emergent cooperative, hybrid racial playstyles and long-term diplomatic mastery.
*/

use crate::race::Race;
use std::collections::HashMap;

/// Represents the diplomatic relationship between two races.
#[derive(Debug, Clone)]
pub struct DiplomacyRelation {
    pub trust: f32, // 0.0 (hostile) to 1.0 (allied)
    pub active_treaties: Vec<String>,
}

impl Default for DiplomacyRelation {
    fn default() -> Self {
        Self {
            trust: 0.35, // Neutral starting point
            active_treaties: Vec::new(),
        }
    }
}

/// Manages all cross-race diplomatic relations for an entity (player or NPC).
#[derive(Debug, Clone)]
pub struct DiplomacyManager {
    pub relations: HashMap<(Race, Race), DiplomacyRelation>,
}

impl DiplomacyManager {
    pub fn new() -> Self {
        Self {
            relations: HashMap::new(),
        }
    }

    /// Returns a normalized key for any pair of races (order-independent).
    fn pair_key(r1: Race, r2: Race) -> (Race, Race) {
        if r1 as u8 <= r2 as u8 {
            (r1, r2)
        } else {
            (r2, r1)
        }
    }

    /// Improves (or damages) the diplomatic trust between two races.
    pub fn improve_relation(&mut self, r1: Race, r2: Race, amount: f32) {
        if r1 == r2 {
            return;
        }
        let key = Self::pair_key(r1, r2);
        let entry = self.relations.entry(key).or_insert_with(DiplomacyRelation::default);
        entry.trust = (entry.trust + amount).clamp(0.0, 1.0);
    }

    /// Returns current trust level between two races (0.0 - 1.0).
    pub fn get_trust(&self, r1: Race, r2: Race) -> f32 {
        if r1 == r2 {
            return 1.0;
        }
        let key = Self::pair_key(r1, r2);
        self.relations.get(&key).map(|r| r.trust).unwrap_or(0.35)
    }

    /// Applies passive diplomacy effects to simulation state.
    /// Called every tick when the entity has multiple races unlocked.
    pub fn apply_diplomacy_effects(
        &self,
        unlocked_races: &[Race],
        global_harmony: &mut f32,
        volatility: &mut f32,
        strength: &mut f32,
    ) {
        if unlocked_races.len() < 2 {
            return;
        }

        let mut total_trust = 0.0;
        let mut pair_count = 0;

        for i in 0..unlocked_races.len() {
            for j in (i + 1)..unlocked_races.len() {
                let trust = self.get_trust(unlocked_races[i], unlocked_races[j]);
                total_trust += trust;
                pair_count += 1;
            }
        }

        if pair_count == 0 {
            return;
        }

        let avg_trust = total_trust / pair_count as f32;

        // High average trust between unlocked races boosts harmony and stability
        if avg_trust > 0.6 {
            *global_harmony = (*global_harmony + 0.025 * avg_trust).min(4.0);
            *volatility = (*volatility - 0.012 * avg_trust).max(0.05);
            *strength = (*strength + 0.015 * avg_trust).min(5.0);
        }

        // Very high diplomacy provides strong repair and backlash resistance
        if avg_trust > 0.85 {
            *volatility = (*volatility - 0.025).max(0.03);
            if *global_harmony > 2.5 {
                *strength = (*strength + 0.04).min(5.5);
            }
        }
    }

    /// Returns a summary string for UI / status reporting.
    pub fn get_diplomacy_summary(&self, unlocked_races: &[Race]) -> String {
        if unlocked_races.len() < 2 {
            return "No cross-race diplomacy active".to_string();
        }

        let mut summary = String::from("Cross-Race Diplomacy: ");
        let mut high_trust_pairs = 0;

        for i in 0..unlocked_races.len() {
            for j in (i + 1)..unlocked_races.len() {
                let trust = self.get_trust(unlocked_races[i], unlocked_races[j]);
                if trust > 0.7 {
                    high_trust_pairs += 1;
                }
            }
        }

        if high_trust_pairs > 0 {
            summary.push_str(&format!("{} strong alliances", high_trust_pairs));
        } else {
            summary.push_str("emerging relations");
        }
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diplomacy_improvement_and_effects() {
        let mut mgr = DiplomacyManager::new();
        mgr.improve_relation(Race::Harmonic, Race::Terran, 0.4);
        assert!(mgr.get_trust(Race::Harmonic, Race::Terran) > 0.7);

        let mut harmony = 1.0f32;
        let mut vol = 0.8f32;
        let mut strg = 1.5f32;
        let races = vec![Race::Harmonic, Race::Terran];
        mgr.apply_diplomacy_effects(&races, &mut harmony, &mut vol, &mut strg);

        assert!(harmony > 1.0);
        assert!(vol < 0.8);
    }
}
