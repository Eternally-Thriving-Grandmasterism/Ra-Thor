//! crates/powrush/src/npc/stats.rs
//! Dynamic stat scaling for all NPCs — mercy-gated, data-driven, template-based
//! v1.0 | TOLC 8 + MIAL/MWPO aligned | AG-SML v1.0

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NpcStats {
    pub health: f32,
    pub max_health: f32,
    pub damage: f32,
    pub defense: f32,
    pub speed: f32,
    pub mercy_valence: f64,
    pub mercy_resistance: f32,
    pub ascension_level: u32,
    pub is_awakened: bool,
}

impl Default for NpcStats {
    fn default() -> Self {
        Self {
            health: 100.0,
            max_health: 100.0,
            damage: 10.0,
            defense: 5.0,
            speed: 5.0,
            mercy_valence: 0.75,
            mercy_resistance: 0.5,
            ascension_level: 0,
            is_awakened: false,
        }
    }
}

impl NpcStats {
    pub fn from_template(template_id: &str) -> Self {
        match template_id {
            "hostile_bandit" | "aggressive" => Self {
                health: 80.0,
                max_health: 80.0,
                damage: 18.0,
                defense: 4.0,
                speed: 6.5,
                mercy_valence: 0.35,
                mercy_resistance: 0.3,
                ascension_level: 0,
                is_awakened: false,
            },
            "merchant" | "neutral" => Self {
                health: 120.0,
                max_health: 120.0,
                damage: 5.0,
                defense: 8.0,
                speed: 4.0,
                mercy_valence: 0.92,
                mercy_resistance: 0.8,
                ascension_level: 1,
                is_awakened: false,
            },
            "companion" | "friendly" => Self {
                health: 110.0,
                max_health: 110.0,
                damage: 12.0,
                defense: 7.0,
                speed: 5.5,
                mercy_valence: 0.88,
                mercy_resistance: 0.7,
                ascension_level: 2,
                is_awakened: true,
            },
            "ambient" | "villager" => Self {
                health: 60.0,
                max_health: 60.0,
                damage: 3.0,
                defense: 2.0,
                speed: 3.5,
                mercy_valence: 0.85,
                mercy_resistance: 0.9,
                ascension_level: 0,
                is_awakened: false,
            },
            _ => Self::default(),
        }
    }

    /// MWPO - Mercy Weighted Personality Override
    pub fn apply_world_mercy(&mut self, world_mercy: f64) {
        let mercy_factor = (world_mercy as f32).clamp(0.0, 1.2);
        if self.mercy_valence < 0.6 {
            self.damage *= 0.7 + 0.3 * mercy_factor;
        }
        self.defense *= 0.9 + 0.2 * mercy_factor;
        self.speed *= 0.95 + 0.1 * mercy_factor;
        self.mercy_valence = (self.mercy_valence * 0.7 + world_mercy * 0.3).clamp(0.0, 1.5);
    }

    pub fn take_damage(&mut self, amount: f32) {
        self.health = (self.health - amount).max(0.0);
    }

    pub fn is_alive(&self) -> bool {
        self.health > 0.0
    }
}