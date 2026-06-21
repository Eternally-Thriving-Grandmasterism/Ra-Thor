/*!
# Powrush MMOARPG Simulator — Cross-Race Diplomacy Module

**Autonomicity Games Sovereign Mercy License (AG-SML) v1.0**  
**Aligned with TOLC 8 Mercy Lattice, Ra-Thor ONE Organism, 13+ PATSAGi Councils**  
**v1.1 — Active Treaty Mechanics**

Enables meaningful diplomatic relations + active treaties between the 5 races.
Treaties (HarmonyAccord, TradePact, ResearchExchange, MutualDefense) provide powerful ongoing bonuses
when signed between unlocked races. High trust + cooperative conditions allow treaties to be signed.

Creates deep long-term cooperative hybrid racial identity and diplomatic mastery.
*/

use crate::race::Race;
use std::collections::HashMap;

/// Treaty type constants
pub const TREATY_HARMONY_ACCORD: &str = "HarmonyAccord";
pub const TREATY_TRADE_PACT: &str = "TradePact";
pub const TREATY_RESEARCH_EXCHANGE: &str = "ResearchExchange";
pub const TREATY_MUTUAL_DEFENSE: &str = "MutualDefense";

/// Represents the diplomatic relationship between two races.
#[derive(Debug, Clone)]
pub struct DiplomacyRelation {
    pub trust: f32, // 0.0 (hostile) to 1.0 (allied)
    pub active_treaties: Vec<String>,
}

impl Default for DiplomacyRelation {
    fn default() -> Self {
        Self {
            trust: 0.35,
            active_treaties: Vec::new(),
        }
    }
}

/// Manages all cross-race diplomatic relations for an entity.
#[derive(Debug, Clone)]
pub struct DiplomacyManager {
    pub relations: HashMap<(Race, Race), DiplomacyRelation>,
}

impl DiplomacyManager {
    pub fn new() -> Self {
        Self { relations: HashMap::new() }
    }

    fn pair_key(r1: Race, r2: Race) -> (Race, Race) {
        if r1 as u8 <= r2 as u8 { (r1, r2) } else { (r2, r1) }
    }

    pub fn improve_relation(&mut self, r1: Race, r2: Race, amount: f32) {
        if r1 == r2 { return; }
        let key = Self::pair_key(r1, r2);
        let entry = self.relations.entry(key).or_insert_with(DiplomacyRelation::default);
        entry.trust = (entry.trust + amount).clamp(0.0, 1.0);
    }

    pub fn get_trust(&self, r1: Race, r2: Race) -> f32 {
        if r1 == r2 { return 1.0; }
        let key = Self::pair_key(r1, r2);
        self.relations.get(&key).map(|r| r.trust).unwrap_or(0.35)
    }

    /// Attempts to sign an active treaty. Requires trust >= 0.65.
    /// Returns true on success (or if already signed).
    pub fn sign_treaty(&mut self, r1: Race, r2: Race, treaty: &str) -> bool {
        if r1 == r2 { return false; }
        let key = Self::pair_key(r1, r2);
        let entry = self.relations.entry(key).or_insert_with(DiplomacyRelation::default);
        if entry.trust < 0.65 { return false; }
        if !entry.active_treaties.contains(&treaty.to_string()) {
            entry.active_treaties.push(treaty.to_string());
            entry.trust = (entry.trust + 0.05).min(1.0); // Signing bonus
        }
        true
    }

    pub fn has_active_treaty(&self, r1: Race, r2: Race, treaty: &str) -> bool {
        if r1 == r2 { return false; }
        let key = Self::pair_key(r1, r2);
        self.relations.get(&key)
            .map(|r| r.active_treaties.contains(&treaty.to_string()))
            .unwrap_or(false)
    }

    pub fn apply_diplomacy_effects(
        &self,
        unlocked_races: &[Race],
        global_harmony: &mut f32,
        volatility: &mut f32,
        strength: &mut f32,
    ) {
        if unlocked_races.len() < 2 { return; }

        let mut total_trust = 0.0;
        let mut pair_count = 0;

        for i in 0..unlocked_races.len() {
            for j in (i + 1)..unlocked_races.len() {
                let trust = self.get_trust(unlocked_races[i], unlocked_races[j]);
                total_trust += trust;
                pair_count += 1;
            }
        }

        if pair_count == 0 { return; }
        let avg_trust = total_trust / pair_count as f32;

        if avg_trust > 0.6 {
            *global_harmony = (*global_harmony + 0.025 * avg_trust).min(4.0);
            *volatility = (*volatility - 0.012 * avg_trust).max(0.05);
            *strength = (*strength + 0.015 * avg_trust).min(5.0);
        }

        if avg_trust > 0.85 {
            *volatility = (*volatility - 0.025).max(0.03);
            if *global_harmony > 2.5 {
                *strength = (*strength + 0.04).min(5.5);
            }
        }
    }

    /// Applies strong ongoing bonuses from any active treaties between unlocked races.
    pub fn apply_treaty_effects(
        &self,
        unlocked_races: &[Race],
        global_harmony: &mut f32,
        volatility: &mut f32,
        strength: &mut f32,
    ) {
        if unlocked_races.len() < 2 { return; }

        for i in 0..unlocked_races.len() {
            for j in (i + 1)..unlocked_races.len() {
                let r1 = unlocked_races[i];
                let r2 = unlocked_races[j];

                if self.has_active_treaty(r1, r2, TREATY_HARMONY_ACCORD) {
                    *global_harmony = (*global_harmony + 0.045).min(4.8);
                    *volatility = (*volatility - 0.018).max(0.02);
                }
                if self.has_active_treaty(r1, r2, TREATY_TRADE_PACT) {
                    *strength = (*strength + 0.03).min(6.0);
                }
                if self.has_active_treaty(r1, r2, TREATY_RESEARCH_EXCHANGE) {
                    if *global_harmony > 2.2 {
                        *strength = (*strength + 0.035).min(6.2);
                    }
                }
                if self.has_active_treaty(r1, r2, TREATY_MUTUAL_DEFENSE) {
                    if *volatility > 0.9 {
                        *volatility = (*volatility - 0.04).max(0.1); // Emergency stabilization
                    }
                }
            }
        }
    }

    pub fn get_diplomacy_summary(&self, unlocked_races: &[Race]) -> String {
        if unlocked_races.len() < 2 {
            return "No cross-race diplomacy active".to_string();
        }

        let mut high_trust_pairs = 0;
        let mut total_treaties = 0;

        for i in 0..unlocked_races.len() {
            for j in (i + 1)..unlocked_races.len() {
                let trust = self.get_trust(unlocked_races[i], unlocked_races[j]);
                if trust > 0.7 { high_trust_pairs += 1; }
                // Count treaties for this pair
                if let Some(rel) = self.relations.get(&Self::pair_key(unlocked_races[i], unlocked_races[j])) {
                    total_treaties += rel.active_treaties.len();
                }
            }
        }

        if total_treaties > 0 {
            format!("Cross-Race Diplomacy: {} strong alliances + {} active treaties", high_trust_pairs, total_treaties)
        } else if high_trust_pairs > 0 {
            format!("Cross-Race Diplomacy: {} strong alliances", high_trust_pairs)
        } else {
            "Cross-Race Diplomacy: emerging relations".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diplomacy_and_treaties() {
        let mut mgr = DiplomacyManager::new();
        mgr.improve_relation(Race::Harmonic, Race::Terran, 0.5);
        assert!(mgr.sign_treaty(Race::Harmonic, Race::Terran, TREATY_HARMONY_ACCORD));
        assert!(mgr.has_active_treaty(Race::Harmonic, Race::Terran, TREATY_HARMONY_ACCORD));

        let mut harmony = 1.0f32;
        let mut vol = 0.8f32;
        let mut strg = 1.5f32;
        let races = vec![Race::Harmonic, Race::Terran];
        mgr.apply_treaty_effects(&races, &mut harmony, &mut vol, &mut strg);
        assert!(harmony > 1.0);
    }
}
