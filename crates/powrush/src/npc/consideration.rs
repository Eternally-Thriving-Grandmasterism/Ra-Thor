//! crates/powrush/src/npc/consideration.rs
//! Consideration trait + concrete Powrush considerations for Utility AI
//! Mercy, Ascension, and Post-Scarcity as first-class inputs | v1.0 | AG-SML v1.0

use super::blackboard::NpcBlackboard;

/// A normalized consideration that returns a score in [0.0, 1.0].
/// Higher scores indicate stronger preference for a particular action or behavior.
pub trait Consideration {
    /// Returns a normalized score in [0.0, 1.0].
    fn score(&self, blackboard: &NpcBlackboard) -> f64;

    /// Human-readable name for debugging and logging.
    fn name(&self) -> &'static str;
}

// ============================================================================
// Concrete Considerations (Powrush + Mercy aligned)
// ============================================================================

/// How aligned the current world/player state is with mercy principles.
/// High mercy alignment → more peaceful / cooperative behavior.
pub struct MercyAlignmentConsideration;

impl Consideration for MercyAlignmentConsideration {
    fn score(&self, blackboard: &NpcBlackboard) -> f64 {
        let mercy = (blackboard.world_mercy + blackboard.player_mercy) / 2.0;
        mercy.clamp(0.0, 1.0)
    }

    fn name(&self) -> &'static str {
        "MercyAlignment"
    }
}

/// Self-preservation: lower health = higher urgency to act defensively or flee.
pub struct HealthConsideration;

impl Consideration for HealthConsideration {
    fn score(&self, blackboard: &NpcBlackboard) -> f64 {
        if blackboard.max_health <= 0.0 {
            return 1.0;
        }
        let health_ratio = (blackboard.current_health / blackboard.max_health) as f64;
        (1.0 - health_ratio).clamp(0.0, 1.0)
    }

    fn name(&self) -> &'static str {
        "HealthUrgency"
    }
}

/// How threatening the player currently appears.
/// Combines proximity (via last known position), detection count, and recent combat.
pub struct PlayerThreatConsideration;

impl Consideration for PlayerThreatConsideration {
    fn score(&self, blackboard: &NpcBlackboard) -> f64 {
        let mut threat = 0.3; // baseline

        if blackboard.last_known_player_position.is_some() {
            threat += 0.25;
        }

        let detection_factor = (blackboard.times_detected_player as f64 * 0.08).min(0.4);
        threat += detection_factor;

        if blackboard.last_combat_time > -100.0 {
            threat += 0.35;
        }

        threat.clamp(0.0, 1.0)
    }

    fn name(&self) -> &'static str {
        "PlayerThreat"
    }
}

/// Post-scarcity state strongly influences NPC behavior.
/// In true post-scarcity, NPCs are less aggressive and more curious / helpful.
pub struct PostScarcityConsideration;

impl Consideration for PostScarcityConsideration {
    fn score(&self, blackboard: &NpcBlackboard) -> f64 {
        if blackboard.is_post_scarcity {
            0.85
        } else {
            0.25
        }
    }

    fn name(&self) -> &'static str {
        "PostScarcityState"
    }
}

/// Player's ascension level. Higher ascension often means more respect,
/// different dialogue options, or changed behavior from NPCs.
pub struct PlayerAscensionConsideration;

impl Consideration for PlayerAscensionConsideration {
    fn score(&self, blackboard: &NpcBlackboard) -> f64 {
        (blackboard.player_ascension / 10.0).clamp(0.0, 1.0)
    }

    fn name(&self) -> &'static str {
        "PlayerAscension"
    }
}