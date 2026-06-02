//! # Powrush Enemy Perception Systems
//!
//! Mercy-aware enemy senses with line of sight and audio propagation.
//!
//! Perception types:
//! - Visual (line of sight + distance)
//! - Auditory (sound propagation with distance/wall dampening)
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
    /// Simple line of sight check (placeholder for real raycasting)
    pub fn has_line_of_sight(
        from_x: f32,
        from_y: f32,
        to_x: f32,
        to_y: f32,
        _map_data: Option<&[u8]>,
    ) -> bool {
        let dx = to_x - from_x;
        let dy = to_y - from_y;
        let dist_sq = dx * dx + dy * dy;
        dist_sq < 400.0
    }

    /// Audio propagation factor (0.0–1.0)
    /// Simulates distance and wall dampening.
    /// In a full implementation, this would use map data for realistic sound travel.
    pub fn audio_propagation_factor(
        distance: f32,
        noise_level: f32,
        _map_data: Option<&[u8]>,
    ) -> f32 {
        // Distance dampening
        let distance_factor = (1.0 - (distance / 30.0)).max(0.0);

        // Simple wall dampening approximation (future: real map-based)
        let wall_dampening = 0.85; // Assume some walls always present

        (distance_factor * noise_level * wall_dampening).clamp(0.0, 1.0)
    }

    /// Main detection check
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
        // Visual (requires LOS)
        if distance <= profile.visual_range {
            if Self::has_line_of_sight(from_x, from_y, to_x, to_y, map_data) {
                return true;
            }
        }

        // Auditory (uses audio propagation)
        let audio_factor = Self::audio_propagation_factor(distance, noise_level, map_data);
        if audio_factor > 0.25 && distance <= profile.auditory_range {
            return true;
        }

        // MercySense
        let mercy_factor = if enemy_behavior == crate::enemy_behavior::EnemyBehavior::Awakened {
            2.0
        } else {
            1.0
        };
        if distance <= profile.mercy_sense_range * mercy_factor {
            return true;
        }

        // Spiritual
        if player_mercy > 0.85 && distance <= profile.spiritual_range {
            return true;
        }

        false
    }
}
