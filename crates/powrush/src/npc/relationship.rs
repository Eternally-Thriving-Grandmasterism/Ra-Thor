//! crates/powrush/src/npc/relationship.rs
//! Production-grade Relationship & Reputation System for Powrush NPCs
//! Mercy, Ascension, and TOLC as first-class modifiers | v1.0 | AG-SML v1.0

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipLevel {
    Hostile = -2,
    Wary = -1,
    Neutral = 0,
    Friendly = 1,
    Revered = 2,
    Devoted = 3,
}

impl RelationshipLevel {
    pub fn from_reputation(rep: i32) -> Self {
        match rep {
            i if i <= -60 => RelationshipLevel::Hostile,
            i if i <= -25 => RelationshipLevel::Wary,
            i if i < 25 => RelationshipLevel::Neutral,
            i if i < 55 => RelationshipLevel::Friendly,
            i if i < 80 => RelationshipLevel::Revered,
            _ => RelationshipLevel::Devoted,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub reputation: i32,           // -100 to +100
    pub level: RelationshipLevel,
    pub last_interaction_time: f32,
    pub times_helped: u32,
    pub times_harmed: u32,
    pub mercy_influence: f64,      // How much mercy has affected this relationship
}

impl Default for Relationship {
    fn default() -> Self {
        Self {
            reputation: 0,
            level: RelationshipLevel::Neutral,
            last_interaction_time: -999.0,
            times_helped: 0,
            times_harmed: 0,
            mercy_influence: 0.0,
        }
    }
}

impl Relationship {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update_level(&mut self) {
        self.level = RelationshipLevel::from_reputation(self.reputation);
    }

    /// Apply a mercy-aligned action modifier
    pub fn apply_mercy_action(&mut self, mercy_impact: f64, is_positive: bool) {
        let change = if is_positive {
            (mercy_impact * 8.0) as i32
        } else {
            (mercy_impact * -12.0) as i32
        };

        self.reputation = (self.reputation + change).clamp(-100, 100);
        self.mercy_influence += mercy_impact * if is_positive { 1.0 } else { -1.5 };
        self.update_level();
    }

    /// Player ascension gives natural reputation bonus over time
    pub fn apply_ascension_influence(&mut self, player_ascension: f64) {
        if player_ascension > 3.0 {
            let bonus = ((player_ascension - 3.0) * 2.0) as i32;
            self.reputation = (self.reputation + bonus).min(100);
            self.update_level();
        }
    }

    /// Post-scarcity worlds make NPCs more open to positive relationships
    pub fn apply_post_scarcity_modifier(&mut self, is_post_scarcity: bool) {
        if is_post_scarcity && self.reputation > -20 {
            self.reputation = (self.reputation + 3).min(100);
            self.update_level();
        }
    }

    pub fn is_hostile(&self) -> bool {
        matches!(self.level, RelationshipLevel::Hostile)
    }

    pub fn is_friendly(&self) -> bool {
        matches!(self.level, RelationshipLevel::Friendly | RelationshipLevel::Revered | RelationshipLevel::Devoted)
    }
}