//! crates/powrush/src/npc/epigenetic.rs
//! Epigenetic Blessing + RBE Distribution System for v15 NPCs
//! Mercy, relationship, and post-scarcity drive blessing flow into Powrush economy
//! ONE Organism + PATSAGi aligned | AG-SML v1.0

use super::NpcBlackboard;

/// Distributes epigenetic blessings based on current blackboard state.
/// In a full RBE implementation this would credit the global resource pool
/// or trigger Powrush item / skill unlocks for the player.
pub fn distribute_epigenetic_blessing(blackboard: &mut NpcBlackboard) -> f64 {
    let mut blessing = 0.0;

    // Base mercy contribution
    if blackboard.current_mercy_valence > 0.75 {
        blessing += (blackboard.current_mercy_valence - 0.75) * 4.0;
    }

    // Relationship bonus
    if blackboard.player_mercy > 0.8 {
        blessing += 1.2;
    }

    // Post-scarcity multiplier (RBE core)
    if blackboard.is_post_scarcity {
        blessing *= 1.6;
    }

    // Cap and record
    blessing = blessing.min(8.0);
    if blessing > 0.5 {
        blackboard.record_event(&format!("Epigenetic blessing distributed: +{:.1}", blessing));
    }

    blessing
}

/// Batch apply blessings across all NPCs in a blackboard list (for simulation tick)
pub fn batch_distribute_blessings(blackboards: &mut [NpcBlackboard]) -> f64 {
    blackboards.iter_mut().map(distribute_epigenetic_blessing).sum()
}