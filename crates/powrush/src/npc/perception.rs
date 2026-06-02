//! crates/powrush/src/npc/perception.rs
//! Perception system that updates the blackboard with multi-sense data
//! Line of sight, audio, mercy-sense | v1.0 | AG-SML v1.0

use super::blackboard::NpcBlackboard;
use nalgebra::Vector2;

pub struct PerceptionSystem;

impl PerceptionSystem {
    pub fn update(blackboard: &mut NpcBlackboard, player_pos: Option<Vector2<f32>>, enemy_pos: Vector2<f32>, noise_level: f32, has_los: bool) {
        blackboard.has_line_of_sight = has_los;
        blackboard.current_noise_level = noise_level;

        if let Some(ppos) = player_pos {
            let dist = (ppos - enemy_pos).norm();
            // Simple audio strength falloff
            blackboard.audio_strength = (noise_level / (dist + 1.0)).min(1.0);
            // Visual strength (requires LOS)
            blackboard.visual_strength = if has_los { (10.0 / (dist + 1.0)).min(1.0) } else { 0.0 };

            if has_los || blackboard.audio_strength > 0.3 {
                blackboard.last_known_player_position = Some(ppos);
                blackboard.last_seen_time = 0.0; // reset
            }
        }

        // Mercy-sense bonus (high mercy players are easier to "feel")
        if blackboard.player_mercy > 0.8 {
            blackboard.visual_strength = (blackboard.visual_strength + 0.2).min(1.0);
        }
    }
}