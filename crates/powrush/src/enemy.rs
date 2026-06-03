//! # Powrush Enemy Stat Scaling System
//!
//! Dynamic enemy stat scaling that keeps combat meaningful across all ascension levels.
//!
//! Scaling factors:
//! - Ascension level (player power)
//! - Mercy compliance (world state)
//! - Collective joy / post-scarcity
//!
//! Higher ascension = stronger enemies (to maintain challenge)
//! High mercy + post-scarcity = potentially "awakened" or less hostile enemies

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnemyStats {
    pub health: f32,
    pub damage: f32,
    pub defense: f32,
    pub speed: f32,
    pub mercy_resistance: f32, // How resistant to mercy-based effects
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnemyTemplate {
    pub name: String,
    pub base_stats: EnemyStats,
    pub difficulty_tier: u32,
}

pub struct EnemyScaler;

impl EnemyScaler {
    /// Scale enemy stats based on player and world state
    pub fn scale_enemy(
        template: &EnemyTemplate,
        player_ascension: f32,   // 0.0–1.0
        world_mercy: f32,        // 0.0–1.0
        collective_joy: f32,
        is_post_scarcity: bool,
    ) -> EnemyStats {
        let mut stats = template.base_stats.clone();

        // Ascension scaling (stronger player = stronger enemies)
        let ascension_mult = 1.0 + (player_ascension * 1.2);

        // Mercy influence (high mercy = slightly easier or different enemies)
        let mercy_mult = 1.0 - (world_mercy * 0.15);

        // Joy / post-scarcity effect
        let joy_mult = 0.95 + (collective_joy / 300.0);
        let scarcity_mult = if is_post_scarcity { 0.9 } else { 1.05 };

        let total_mult = ascension_mult * mercy_mult * joy_mult * scarcity_mult;

        stats.health *= total_mult;
        stats.damage *= total_mult * 0.95; // Damage scales slightly less aggressively
        stats.defense *= total_mult * 0.9;
        stats.speed *= 1.0 + (player_ascension * 0.3); // Faster at high ascension

        // Mercy resistance decreases in high-mercy worlds
        stats.mercy_resistance *= 1.0 - (world_mercy * 0.4);

        stats
    }
}
