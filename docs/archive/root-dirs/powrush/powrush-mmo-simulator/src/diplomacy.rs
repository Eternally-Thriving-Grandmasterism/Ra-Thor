/*!
# Powrush MMOARPG Simulator — Cross-Race Diplomacy Module

**Autonomicity Games Sovereign Mercy License (AG-SML) v1.0**  
**Aligned with TOLC 8 Mercy Lattice, Ra-Thor ONE Organism, 13+ PATSAGi Councils**  
**v1.4 — Treaty Renewal Mechanics**

Enables meaningful diplomatic relations + active treaties between the 5 races.
Treaties now have **finite lifetimes**, automatically expire unless renewed, and support full **player-initiated renewal**.

High-harmony hybrid builds are incentivized to actively maintain and renew diplomatic relations for continuous powerful bonuses.
Player proposals, pending acceptance, active treaty effects, and expiration cleanup remain fully supported.
*/

use crate::race::Race;
use std::collections::HashMap;

/// Treaty type constants
pub const TREATY_HARMONY_ACCORD: &str = "HarmonyAccord";
pub const TREATY_TRADE_PACT: &str = "TradePact";
pub const TREATY_RESEARCH_EXCHANGE: &str = "ResearchExchange";
pub const TREATY_MUTUAL_DEFENSE: &str = "MutualDefense";

/// Default treaty durations in simulation ticks
pub const TREATY_DURATION_DEFAULT: u64 = 1200;      // ~20 minutes of simulation time
pub const TREATY_DURATION_HARMONY: u64 = 1800;     // Longer for core cooperative treaties
pub const TREATY_DURATION_TRADE: u64 = 900;
pub const TREATY_DURATION_RESEARCH: u64 = 1100;
pub const TREATY_DURATION_DEFENSE: u64 = 800;

/// Represents one active treaty with its expiration tick.
#[derive(Debug, Clone)]
pub struct ActiveTreaty {
    pub treaty_type: String,
    pub expires_at_tick: u64,
}

impl ActiveTreaty {
    pub fn is_expired(&self, current_tick: u64) -> bool {
        current_tick >= self.expires_at_tick
    }
}

/// Represents the diplomatic relationship between two races.
#[derive(Debug, Clone)]
pub struct DiplomacyRelation {
    pub trust: f32, // 0.0 (hostile) to 1.0 (allied)
    pub active_treaties: Vec<ActiveTreaty>,
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

    /// Returns the duration for a given treaty type.
    fn get_treaty_duration(treaty: &str) -> u64 {
        match treaty {
            TREATY_HARMONY_ACCORD => TREATY_DURATION_HARMONY,
            TREATY_TRADE_PACT => TREATY_DURATION_TRADE,
            TREATY_RESEARCH_EXCHANGE => TREATY_DURATION_RESEARCH,
            TREATY_MUTUAL_DEFENSE => TREATY_DURATION_DEFENSE,
            _ => TREATY_DURATION_DEFAULT,
        }
    }

    /// Player-initiated treaty proposal.
    /// Requires trust >= 0.55 to propose. Does not sign immediately.
    pub fn propose_treaty(&mut self, r1: Race, r2: Race, treaty: &str) -> bool {
        if r1 == r2 { return false; }
        let key = Self::pair_key(r1, r2);
        let trust = self.get_trust(r1, r2);
        if trust < 0.55 { return false; }

        let entry = self.relations.entry(key).or_insert_with(DiplomacyRelation::default);
        if entry.active_treaties.iter().any(|t| t.treaty_type == treaty) {
            return true; // Already active
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
    /// Creates an ActiveTreaty with proper expiration time.
    pub fn accept_pending_treaty(&mut self, r1: Race, r2: Race, treaty: &str, current_tick: u64) -> bool {
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
                if !entry.active_treaties.iter().any(|t| t.treaty_type == treaty) {
                    let duration = Self::get_treaty_duration(treaty);
                    entry.active_treaties.push(ActiveTreaty {
                        treaty_type: treaty.to_string(),
                        expires_at_tick: current_tick + duration,
                    });
                    entry.trust = (entry.trust + 0.06).min(1.0);
                }
                return true;
            }
        }
        false
    }

    /// Signs a treaty directly (legacy/auto path). Requires trust >= 0.65.
    pub fn sign_treaty(&mut self, r1: Race, r2: Race, treaty: &str, current_tick: u64) -> bool {
        if r1 == r2 { return false; }
        let key = Self::pair_key(r1, r2);
        let entry = self.relations.entry(key).or_insert_with(DiplomacyRelation::default);
        if entry.trust < 0.65 { return false; }

        if !entry.active_treaties.iter().any(|t| t.treaty_type == treaty) {
            let duration = Self::get_treaty_duration(treaty);
            entry.active_treaties.push(ActiveTreaty {
                treaty_type: treaty.to_string(),
                expires_at_tick: current_tick + duration,
            });
            entry.trust = (entry.trust + 0.05).min(1.0);
        }
        true
    }

    /// Player-initiated treaty renewal.
    /// Extends the expiration timer of an active (non-expired) treaty by its full duration.
    /// Requires trust >= 0.60. Grants a small trust bonus on success.
    /// Returns true if the treaty was successfully renewed.
    pub fn renew_treaty(&mut self, r1: Race, r2: Race, treaty: &str, current_tick: u64) -> bool {
        if r1 == r2 { return false; }
        let key = Self::pair_key(r1, r2);
        let trust = self.get_trust(r1, r2);
        if trust < 0.60 { return false; }

        if let Some(rel) = self.relations.get_mut(&key) {
            if let Some(treaty_entry) = rel.active_treaties.iter_mut().find(|t| t.treaty_type == treaty && !t.is_expired(current_tick)) {
                let duration = Self::get_treaty_duration(treaty);
                treaty_entry.expires_at_tick = current_tick + duration; // Full duration reset
                rel.trust = (rel.trust + 0.04).min(1.0);

                // Clear any pending proposal for this treaty
                if let Some(pending) = self.pending_proposals.get_mut(&key) {
                    pending.retain(|t| t != treaty);
                    if pending.is_empty() {
                        self.pending_proposals.remove(&key);
                    }
                }
                return true;
            }
        }
        false
    }

    pub fn has_active_treaty(&self, r1: Race, r2: Race, treaty: &str, current_tick: u64) -> bool {
        if r1 == r2 { return false; }
        let key = Self::pair_key(r1, r2);
        self.relations.get(&key)
            .map(|r| {
                r.active_treaties.iter().any(|t| t.treaty_type == treaty && !t.is_expired(current_tick))
            })
            .unwrap_or(false)
    }

    /// Removes all expired treaties from all relations.
    /// Applies a small trust penalty for letting treaties lapse (encourages renewal).
    pub fn cleanup_expired_treaties(&mut self, current_tick: u64) {
        for relation in self.relations.values_mut() {
            let before = relation.active_treaties.len();
            relation.active_treaties.retain(|t| !t.is_expired(current_tick));
            let expired_count = before - relation.active_treaties.len();

            if expired_count > 0 {
                // Small trust penalty for expired treaties (encourages diplomatic maintenance)
                relation.trust = (relation.trust - 0.03 * expired_count as f32).max(0.1);
            }
        }
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

    /// Applies strong ongoing bonuses from any NON-EXPIRED active treaties.
    pub fn apply_treaty_effects(
        &self,
        unlocked_races: &[Race],
        global_harmony: &mut f32,
        volatility: &mut f32,
        strength: &mut f32,
        current_tick: u64,
    ) {
        if unlocked_races.len() < 2 { return; }

        for i in 0..unlocked_races.len() {
            for j in (i + 1)..unlocked_races.len() {
                let r1 = unlocked_races[i];
                let r2 = unlocked_races[j];
                let key = Self::pair_key(r1, r2);

                if let Some(rel) = self.relations.get(&key) {
                    for treaty in &rel.active_treaties {
                        if treaty.is_expired(current_tick) { continue; }

                        match treaty.treaty_type.as_str() {
                            TREATY_HARMONY_ACCORD => {
                                *global_harmony = (*global_harmony + 0.045).min(4.8);
                                *volatility = (*volatility - 0.018).max(0.02);
                            }
                            TREATY_TRADE_PACT => {
                                *strength = (*strength + 0.03).min(6.0);
                            }
                            TREATY_RESEARCH_EXCHANGE => {
                                if *global_harmony > 2.2 {
                                    *strength = (*strength + 0.035).min(6.2);
                                }
                            }
                            TREATY_MUTUAL_DEFENSE => {
                                if *volatility > 0.9 {
                                    *volatility = (*volatility - 0.04).max(0.1);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    pub fn get_diplomacy_summary(&self, unlocked_races: &[Race], current_tick: u64) -> String {
        if unlocked_races.len() < 2 {
            return "No cross-race diplomacy active".to_string();
        }

        let mut high_trust_pairs = 0;
        let mut total_active_treaties = 0;
        let mut pending_count = 0;
        let mut expiring_soon = 0;

        for i in 0..unlocked_races.len() {
            for j in (i + 1)..unlocked_races.len() {
                let trust = self.get_trust(unlocked_races[i], unlocked_races[j]);
                if trust > 0.7 { high_trust_pairs += 1; }

                if let Some(rel) = self.relations.get(&Self::pair_key(unlocked_races[i], unlocked_races[j])) {
                    for t in &rel.active_treaties {
                        if !t.is_expired(current_tick) {
                            total_active_treaties += 1;
                            if t.expires_at_tick.saturating_sub(current_tick) < 300 {
                                expiring_soon += 1;
                            }
                        }
                    }
                }

                if let Some(pending) = self.pending_proposals.get(&Self::pair_key(unlocked_races[i], unlocked_races[j])) {
                    pending_count += pending.len();
                }
            }
        }

        if total_active_treaties > 0 {
            if pending_count > 0 {
                format!("Cross-Race Diplomacy: {} strong alliances + {} active treaties ({} expiring soon) + {} pending", high_trust_pairs, total_active_treaties, expiring_soon, pending_count)
            } else {
                format!("Cross-Race Diplomacy: {} strong alliances + {} active treaties ({} expiring soon)", high_trust_pairs, total_active_treaties, expiring_soon)
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
    fn test_treaty_expiration_and_cleanup() {
        let mut mgr = DiplomacyManager::new();
        mgr.improve_relation(Race::Harmonic, Race::Terran, 0.7);

        // Sign a short treaty
        assert!(mgr.sign_treaty(Race::Harmonic, Race::Terran, TREATY_HARMONY_ACCORD, 100));
        assert!(mgr.has_active_treaty(Race::Harmonic, Race::Terran, TREATY_HARMONY_ACCORD, 100));

        // At tick 100 + duration it should still be active just before expiry
        let duration = TREATY_DURATION_HARMONY;
        assert!(mgr.has_active_treaty(Race::Harmonic, Race::Terran, TREATY_HARMONY_ACCORD, 100 + duration - 1));

        // After expiry it should be gone
        mgr.cleanup_expired_treaties(100 + duration + 10);
        assert!(!mgr.has_active_treaty(Race::Harmonic, Race::Terran, TREATY_HARMONY_ACCORD, 100 + duration + 10));
    }

    #[test]
    fn test_treaty_renewal() {
        let mut mgr = DiplomacyManager::new();
        mgr.improve_relation(Race::Harmonic, Race::Terran, 0.75);

        assert!(mgr.sign_treaty(Race::Harmonic, Race::Terran, TREATY_HARMONY_ACCORD, 200));
        let original_expiry = mgr.relations.get(&(Race::Harmonic, Race::Terran))
            .unwrap().active_treaties[0].expires_at_tick;

        // Advance time close to expiry
        let duration = TREATY_DURATION_HARMONY;
        let near_expiry = 200 + duration - 50;

        // Renew should succeed and extend the timer
        assert!(mgr.renew_treaty(Race::Harmonic, Race::Terran, TREATY_HARMONY_ACCORD, near_expiry));

        let new_expiry = mgr.relations.get(&(Race::Harmonic, Race::Terran))
            .unwrap().active_treaties[0].expires_at_tick;
        assert!(new_expiry > original_expiry);
        assert!(mgr.get_trust(Race::Harmonic, Race::Terran) > 0.75); // trust bonus applied
    }
}
