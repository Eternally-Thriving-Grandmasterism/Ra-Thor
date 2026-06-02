//! # Powrush Enemy Perception Systems
//!
//! Mercy-aware enemy senses with line of sight checks.
//!
//! Perception types:
//! - Visual (requires line of sight + distance)
//! - Auditory (sound-based, ignores walls)
//! - MercySense (spiritual detection)
//! - Spiritual (high-mercy / awakened enemies)

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptionType {
    Visual,
    Auditory,
    MercySense,
    Spiritual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptionProfile {
    pub visual_range: f32,
    pub auditory_range: f32,
    pub mercy_sense_range: f32,
    pub spiritual_range: f32,
}

impl Default for PerceptionProfile {
    fn default() -> Self {
        Self {
            visual_range: 12.0,
            auditory_range: 8.0,
            mercy_sense_range: 6.0,
            spiritual_range: 4.0,
        }
    }
}

pub struct PerceptionSystem;

impl PerceptionSystem {
    /// Check if there is line of sight between two points.
    /// Placeholder for future raycasting / tile-based LOS.
    pub fn has_line_of_sight(
        from_x: f32,
        from_y: f32,
        to_x: f32,
        to_y: f32,
        _map_data: Option<&[u8]>, // Future: real map/collision data
    ) -> bool {
        // TODO: Implement proper raycasting against map tiles
        let dx = to_x - from_x;
        let dy = to_y - from_y;
        let dist_sq = dx * dx + dy * dy;

        // Simple approximation for now
        dist_sq < 400.0
    }

    /// Main detection check with line of sight for visual perception
    pub fn can_detect_player(
        profile: &PerceptionProfile,
        distance: f32,
        from_x: f32,
        from_y: f32,
        to_x: f32,
        to_y: f32,
        player_mercy: f32,
        enemy_behavior: crate::enemy_behavior::EnemyBehavior,
        noise_level: f32,
        map_data: Option<&[u8]>,
    ) -> bool {
        // Visual detection now requires line of sight
        if distance <= profile.visual_range {
            if Self::has_line_of_sight(from_x, from_y, to_x, to_y, map_data) {
                return true;
            }
        }

        // Auditory detection (ignores walls)
        if noise_level > 0.3 && distance <= profile.auditory_range {
            return true;
        }

        // MercySense (spiritual, largely ignores obstacles)
        let mercy_factor = if enemy_behavior == crate::enemy_behavior::EnemyBehavior::Awakened {
            2.0
        } else {
            1.0
        };
        if distance <= profile.mercy_sense_range * mercy_factor {
            return true;
        }

        // Spiritual perception
        if player_mercy > 0.85 && distance <= profile.spiritual_range {
            return true;
        }

        false
    }
}
