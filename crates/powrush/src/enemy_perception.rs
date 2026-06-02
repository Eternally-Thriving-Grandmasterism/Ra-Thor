//! # Powrush Enemy Perception Systems
//!
//! Mercy-aware enemy senses and detection.
//!
//! Perception types:
//! - Visual (line of sight + distance)
//! - Auditory (sound / movement detection)
//! - MercySense (detects player mercy alignment)
//! - Spiritual (high-mercy / awakened enemies)
//!
//! High-mercy enemies may perceive differently (more "spiritual" awareness).

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
    /// Check if enemy can detect the player
    pub fn can_detect_player(
        profile: &PerceptionProfile,
        distance: f32,
        player_mercy: f32,
        enemy_behavior: crate::enemy_behavior::EnemyBehavior,
        noise_level: f32, // 0.0–1.0
    ) -> bool {
        // Visual detection
        if distance <= profile.visual_range {
            return true;
        }

        // Auditory detection
        if noise_level > 0.3 && distance <= profile.auditory_range {
            return true;
        }

        // MercySense (stronger in high-mercy players or awakened enemies)
        let mercy_factor = if enemy_behavior == crate::enemy_behavior::EnemyBehavior::Awakened {
            2.0
        } else {
            1.0
        };
        if distance <= profile.mercy_sense_range * mercy_factor {
            return true;
        }

        // Spiritual perception (very high mercy worlds)
        if player_mercy > 0.85 && distance <= profile.spiritual_range {
            return true;
        }

        false
    }
}
