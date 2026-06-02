//! # Player System
//!
//! Every sentient being in Powrush is represented by a `Player`.
//!
//! This module tracks the living state of each participant:
//! - Happiness and multi-dimensional needs (tied to the **5-Gene Joy Tetrad**)
//! - Faction loyalty and identity
//! - Ascension progress
//! - Mercy compliance history (passes vs violations)
//!
//! The `Player` is the core unit through which **TOLC principles**, mercy gating,
//! and RBE abundance mechanics are experienced at the individual level.

use crate::faction::Faction;
use crate::ascension::AscensionLevel;
use crate::mercy::MercyGateStatus;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Player {
    pub id: u64,
    pub name: String,
    pub faction: Faction,
    pub happiness: f32,
    pub needs: PlayerNeeds,
    pub ascension_level: AscensionLevel,
    pub resources_owned: std::collections::HashMap<String, f64>,
    pub last_mercy_check: DateTime<Utc>,
    pub total_mercy_passes: u64,
    pub total_mercy_violations: u64,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerNeeds {
    pub food: f32,
    pub water: f32,
    pub energy: f32,
    pub knowledge: f32,
    pub social: f32,
    pub purpose: f32,
    pub joy: f32,                    // Directly tied to 5-Gene Joy Tetrad
}

impl Player {
    pub fn new(name: String, faction: Faction) -> Self {
        let now = Utc::now();
        Self {
            id: rand::random::<u64>(),
            name,
            faction,
            happiness: 75.0,
            needs: PlayerNeeds {
                food: 80.0,
                water: 80.0,
                energy: 70.0,
                knowledge: 60.0,
                social: 65.0,
                purpose: 70.0,
                joy: 68.0,
            },
            ascension_level: AscensionLevel::Seeker,
            resources_owned: std::collections::HashMap::new(),
            last_mercy_check: now,
            total_mercy_passes: 0,
            total_mercy_violations: 0,
            created_at: now,
        }
    }

    /// Update happiness and needs based on world abundance (RBE logic).
    /// Joy and purpose receive strong boosts from abundance, reflecting post-scarcity design.
    pub fn update_happiness_and_needs(&mut self, world_abundance: f64) {
        let abundance_factor = (world_abundance / 10000.0).min(1.5) as f32;

        self.needs.food = (self.needs.food - 3.0 + abundance_factor * 8.0).clamp(0.0, 100.0);
        self.needs.water = (self.needs.water - 2.5 + abundance_factor * 7.5).clamp(0.0, 100.0);
        self.needs.energy = (self.needs.energy - 4.0 + abundance_factor * 9.0).clamp(0.0, 100.0);
        self.needs.knowledge = (self.needs.knowledge + abundance_factor * 6.0).clamp(0.0, 100.0);
        self.needs.social = (self.needs.social + abundance_factor * 5.0).clamp(0.0, 100.0);
        self.needs.purpose = (self.needs.purpose + abundance_factor * 7.0).clamp(0.0, 100.0);
        self.needs.joy = (self.needs.joy + abundance_factor * 8.5).clamp(0.0, 100.0);

        let needs_avg = (self.needs.food + self.needs.water + self.needs.energy +
                         self.needs.knowledge + self.needs.social + self.needs.purpose + self.needs.joy) / 7.0;

        self.happiness = (needs_avg * 0.85 + (self.total_mercy_passes as f32 * 0.15)).clamp(0.0, 100.0);
    }

    /// Apply mercy gate result to this player.
    /// Passed = happiness + joy boost. Failed = significant penalty.
    pub fn apply_mercy_result(&mut self, status: MercyGateStatus) {
        match status {
            MercyGateStatus::Passed => {
                self.total_mercy_passes += 1;
                self.happiness = (self.happiness + 4.0).min(100.0);
                self.needs.joy = (self.needs.joy + 6.0).min(100.0);
            }
            MercyGateStatus::Failed => {
                self.total_mercy_violations += 1;
                self.happiness = (self.happiness - 12.0).max(0.0);
                self.needs.joy = (self.needs.joy - 15.0).max(0.0);
            }
        }
        self.last_mercy_check = Utc::now();
    }

    /// Check if player meets basic ascension criteria.
    pub fn can_ascend(&self) -> bool {
        self.happiness >= 92.0 &&
        self.needs.joy >= 90.0 &&
        self.total_mercy_passes >= 50
    }
}