/*!
# Powrush MMOARPG Simulator — Cross-Race Diplomacy Module

**Autonomicity Games Sovereign Mercy License (AG-SML) v1.0**  
**Aligned with TOLC 8 Mercy Lattice, Ra-Thor ONE Organism, 13+ PATSAGi Councils**  
**v1.2 — Player Initiated Treaty Proposals**

Enables meaningful diplomatic relations + active treaties between the 5 races.
Players (or the simulator on behalf of hybrid builds) can now **initiate treaty proposals**.
Proposals require moderate trust (0.55+) to propose.
High trust (0.65+) allows acceptance into active treaties with powerful ongoing bonuses.

Creates deep long-term cooperative hybrid racial identity, diplomatic mastery, and player-driven treaty negotiation.
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
    pub pending_proposals: HashMap<(Race, Race), Vec<String>>,
}

impl DiplomacyManager {
    pub fn new() -> Self {
        Self {
            relations: HashMap::new(),
            pending_proposals: HashMap::new(),
        }
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

    /// Player-initiated treaty proposal.
    /// Requires trust >= 0.55 to propose. Does not sign immediately.
    /// Returns true if proposal was added (or already existed).
    pub fn propose_treaty(&mut self, r1: Race, r2: Race, treaty: &str) -> bool {
        if r1 == r2 { return false; }
        let key = Self::pair_key(r1, r2);
        let trust = self.get_trust(r1, r2);
        if trust < 0.55 { return false; } // Not enough trust to even propose

        let entry = self.relations.entry(key).or_insert_with(DiplomacyRelation::default);
        // Do not propose if already active
        if entry.active_treaties.contains(&treaty.to_string()) {
            return true;
        }

        let pending = self.pending_proposals.entry(key).or_insert_with(Vec::new);
        if !pending.contains(&treaty.to_string()) {
            pending.push(treaty.to_string());
        }
        true
    }

    pub fn has_pending_proposal(&self, r1: Race, r2: Race, treaty: &str) -> bool {
        if r1 == r2 { return false; }
        let key = Self::pair_key(r1, r2);
        self.pending_proposals.get(&key)
            .map(|p| p.contains(&treaty.to_string()))
            .unwrap_or(false)
    }

    /// Accepts a pending proposal if trust is now high enough (>= 0.65).
    /// Moves it to active_treaties and grants signing bonus.
    pub fn accept_pending_treaty(&mut self, r1: Race, r2: Race, treaty: &str) -> bool {
        if r1 == r2 { return false; }
        let key = Self::pair_key(r1, r2);
        let trust = self.get_trust(r1, r2);
        if trust < 0.65 { return false; }

        if let Some(pending) = self.pending_proposals.get_mut(&key) {
            if let Some(pos) = pending.iter().position(|t| t == treaty) {
                pending.remove(pos);
                if pending.is_empty() {
                    self.pending_proposals.remove(&key);
                }

                let entry = self.relations.entry(key).or_insert_with(DiplomacyRelation::default);
                if !entry.active_treaties.contains(&treaty.to_string()) {
                    entry.active_treaties.push(treaty.to_string());
                    entry.trust = (entry.trust + 0.06).min(1.0); // Slightly higher signing bonus for accepted proposals
                }
                return true;
            }
        }
        false
    }

    /// Attempts to sign an active treaty directly (legacy path). Requires trust >= 0.65.
    pub fn sign_treaty(&mut self, r1: Race, r2: Race, treaty: &str) -> bool {
        if r1 == r2 { return false; }
        let key = Self::pair_key(r1, r2);
        let entry = self.relations.entry(key).or_insert_with(DiplomacyRelation::default);
        if entry.trust < 0.65 { return false; }
        if !entry.active_treaties.contains(&treaty.to_string()) {
            entry.active_treaties.push(treaty.to_string());
            entry.trust = (entry.trust + 0.05).min(1.0);
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
                        *volatility = (*volatility - 0.04).max(0.1);
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
        let mut pending_count = 0;

        for i in 0..unlocked_races.len() {
            for j in (i + 1)..unlocked_races.len() {
                let trust = self.get_trust(unlocked_races[i], unlocked_races[j]);
                if trust > 0.7 { high_trust_pairs += 1; }
                if let Some(rel) = self.relations.get(&Self::pair_key(unlocked_races[i], unlocked_races[j])) {
                    total_treaties += rel.active_treaties.len();
                }
                if let Some(pending) = self.pending_proposals.get(&Self::pair_key(unlocked_races[i], unlocked_races[j])) {
                    pending_count += pending.len();
                }
            }
        }

        if total_treaties > 0 {
            if pending_count > 0 {
                format!("Cross-Race Diplomacy: {} strong alliances + {} active treaties + {} pending proposals", high_trust_pairs, total_treaties, pending_count)
            } else {
                format!("Cross-Race Diplomacy: {} strong alliances + {} active treaties", high_trust_pairs, total_treaties)
            }
        } else if pending_count > 0 {
            format!("Cross-Race Diplomacy: {} strong alliances + {} pending proposals", high_trust_pairs, pending_count)
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
    fn test_player_initiated_treaty_proposals() {
        let mut mgr = DiplomacyManager::new();
        mgr.improve_relation(Race::Harmonic, Race::Terran, 0.6); // Enough to propose
        assert!(mgr.propose_treaty(Race::Harmonic, Race::Terran, TREATY_HARMONY_ACCORD));
        assert!(mgr.has_pending_proposal(Race::Harmonic, Race::Terran, TREATY_HARMONY_ACCORD));

        // Still cannot accept yet (trust 0.6 < 0.65)
        assert!(!mgr.accept_pending_treaty(Race::Harmonic, Race::Terran, TREATY_HARMONY_ACCORD));

        mgr.improve_relation(Race::Harmonic, Race::Terran, 0.1); // Now 0.7
        assert!(mgr.accept_pending_treaty(Race::Harmonic, Race::Terran, TREATY_HARMONY_ACCORD));
        assert!(mgr.has_active_treaty(Race::Harmonic, Race::Terran, TREATY_HARMONY_ACCORD));
        assert!(!mgr.has_pending_proposal(Race::Harmonic, Race::Terran, TREATY_HARMONY_ACCORD));
    }
}
