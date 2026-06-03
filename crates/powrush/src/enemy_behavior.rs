//! # Powrush Enemy Behavior Trees
//!
//! Mercy-aware, state-driven enemy AI behaviors.
//!
//! Behaviors adapt based on:
//! - Player mercy compliance
//! - World post-scarcity state
//! - Player ascension level
//!
//! High mercy + post-scarcity worlds = more passive or "awakened" enemy behavior

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnemyBehavior {
    Aggressive,     // Standard hostile behavior
    Defensive,      // More cautious, higher defense focus
    Passive,        // Low aggression, may flee or ignore
    Awakened,       // High-mercy world behavior (curious, less hostile)
    Desperate,      // Low mercy / scarcity worlds (more aggressive)
}

pub struct BehaviorTree;

impl BehaviorTree {
    /// Determine current behavior based on world and player state
    pub fn determine_behavior(
        player_mercy: f32,
        world_mercy: f32,
        is_post_scarcity: bool,
        player_ascension: f32,
    ) -> EnemyBehavior {
        if is_post_scarcity && world_mercy > 0.85 {
            return EnemyBehavior::Awakened;
        }

        if player_mercy < 0.4 || world_mercy < 0.35 {
            return EnemyBehavior::Desperate;
        }

        if player_ascension > 0.75 && world_mercy > 0.7 {
            return EnemyBehavior::Defensive;
        }

        if player_mercy > 0.75 && world_mercy > 0.65 {
            return EnemyBehavior::Passive;
        }

        EnemyBehavior::Aggressive
    }

    /// Get aggression multiplier for current behavior
    pub fn aggression_multiplier(behavior: EnemyBehavior) -> f32 {
        match behavior {
            EnemyBehavior::Aggressive => 1.0,
            EnemyBehavior::Defensive => 0.7,
            EnemyBehavior::Passive => 0.35,
            EnemyBehavior::Awakened => 0.25,
            EnemyBehavior::Desperate => 1.35,
        }
    }
}
