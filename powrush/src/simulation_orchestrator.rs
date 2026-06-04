//! Weighted Trust Reconciliation for Powrush Entity State
//!
//! Advanced reconciliation that weights incoming state based on the trust
//! level of the source shard. Aligns with mercy-gated, thriving principles.

use crate::entity::SovereignEntity;

/// Trust level of a shard (can be expanded with reputation, PATSAGi alignment, etc.)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShardTrust {
    Low = 0,
    Medium = 1,
    High = 2,
    Sovereign = 3, // Highest trust (e.g. native authoritative shards)
}

impl ShardTrust {
    pub fn weight(&self) -> f32 {
        match self {
            ShardTrust::Low => 0.3,
            ShardTrust::Medium => 0.6,
            ShardTrust::High => 0.85,
            ShardTrust::Sovereign => 0.95,
        }
    }
}

/// Weighted trust reconciliation
/// `incoming_trust` should come from ShardTrust::weight() of the source shard.
pub fn reconcile_entity_state_weighted(
    local: &mut SovereignEntity,
    incoming: &SovereignEntity,
    incoming_trust: f32, // 0.0 - 1.0
) {
    let trust = incoming_trust.clamp(0.0, 1.0);
    let local_weight = 1.0 - trust;

    // Contributions remain largely additive (RBE core)
    local.contributions += (incoming.contributions as f32 * trust) as u64;

    // Skills: weighted blend toward higher value
    for (skill, &incoming_prof) in &incoming.skills {
        let current = local.skills.get(skill).copied().unwrap_or(0.0);
        let blended = current * local_weight + incoming_prof * trust;
        local.skills.insert(skill.clone(), blended.max(current).max(incoming_prof));
    }

    // Valence: weighted blend with mercy bias toward positive change
    let valence_delta = incoming.valence - local.valence;
    if valence_delta > 0.0 {
        local.valence = local.valence * local_weight + incoming.valence * trust;
    } else {
        // Only apply negative change if incoming has very high trust
        if trust > 0.8 {
            local.valence = local.valence * local_weight + incoming.valence * trust;
        }
    }

    // Timestamp
    if incoming.last_active_unix > local.last_active_unix {
        local.last_active_unix = incoming.last_active_unix;
    }

    // Mercy smoothing
    local.apply_valence_update(0.015 * trust);
}

/// Convenience function that uses ShardTrust enum
pub fn reconcile_with_trust_level(
    local: &mut SovereignEntity,
    incoming: &SovereignEntity,
    trust_level: ShardTrust,
) {
    reconcile_entity_state_weighted(local, incoming, trust_level.weight());
}
