//! crates/powrush/src/npc/perception.rs
//! Multi-sense Perception System for v15 hybrid NPC AI
//! Line of Sight + Audio + MercySense | v1.0

use super::blackboard::NpcBlackboard;
use nalgebra::Vector2;

pub type Position = Vector2<f32>;

pub struct PerceptionSystem;

impl PerceptionSystem {
    pub fn new() -> Self { Self }

    /// Updates the blackboard with perception data.
    pub fn update(
        &self,
        blackboard: &mut NpcBlackboard,
        my_position: Position,
        player_position: Option<Position>,
        noise_level: f32,
        dt: f32,
    ) {
        blackboard.current_noise_level = noise_level;

        if let Some(player_pos) = player_position {
            let distance = (player_pos - my_position).magnitude();

            // Simple line of sight check (placeholder for real raycasting)
            blackboard.has_line_of_sight = distance < 20.0;

            if blackboard.has_line_of_sight {
                blackboard.last_known_player_position = Some(player_pos);
                blackboard.last_seen_time = 0.0;
                blackboard.visual_strength = (20.0 - distance) / 20.0;
            }

            // Audio detection
            if noise_level > 0.3 && distance < 25.0 {
                blackboard.audio_strength = (25.0 - distance) / 25.0;
            }
        }

        // Mercy-sense bonus (high mercy players are easier to detect spiritually)
        if blackboard.player_mercy > 0.7 {
            blackboard.visual_strength = (blackboard.visual_strength + 0.2).min(1.0);
        }

        blackboard.last_seen_time += dt;
    }
}