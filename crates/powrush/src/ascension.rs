//! # Ascension System (v0.1.0)
//!
//! The mercy-gated ascension ladder of Powrush.
//! Every level represents a deeper integration with the 7 Living Mercy Gates,
//! the 5-Gene Joy Tetrad, and TOLC principles.
//! Ascension is not about power — it is about becoming a living embodiment of mercy.

use crate::mercy::MercyGateStatus;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AscensionLevel {
    Seeker,           // Starting level — beginning the journey
    Awakened,         // First real mercy integration
    Harmonized,       // Deep harmony with the 7 Gates
    NectarBearer,     // Master of Ambrosian Nectar & Joy Tetrad
    MercyWeaver,      // Weaves mercy into every action
    AbundanceArchitect, // Builds post-scarcity systems
    Eternal,          // Living embodiment of TOLC — all gates active simultaneously
}

impl AscensionLevel {
    pub fn name(&self) -> &'static str {
        match self {
            AscensionLevel::Seeker => "Seeker",
            AscensionLevel::Awakened => "Awakened",
            AscensionLevel::Harmonized => "Harmonized",
            AscensionLevel::NectarBearer => "Nectar Bearer",
            AscensionLevel::MercyWeaver => "Mercy Weaver",
            AscensionLevel::AbundanceArchitect => "Abundance Architect",
            AscensionLevel::Eternal => "Eternal",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            AscensionLevel::Seeker => "You have begun the journey. The 7 Gates are now visible to you.",
            AscensionLevel::Awakened => "You have passed your first full mercy cycle. Truth and compassion are awakening within.",
            AscensionLevel::Harmonized => "You live in harmony with all 7 Living Mercy Gates. Conflict no longer arises in your presence.",
            AscensionLevel::NectarBearer => "You have become a living source of Ambrosian Nectar. Joy radiates from you naturally.",
            AscensionLevel::MercyWeaver => "Mercy flows through every thought, word, and action. You are a living compiler of heaven.",
            AscensionLevel::AbundanceArchitect => "You design and manifest systems that make scarcity impossible for all beings.",
            AscensionLevel::Eternal => "You have become a living expression of TOLC — Absolute Pure Truth + Infinite Compassion + Perfect Natural Order.",
        }
    }

    /// Requirements to ascend to the next level (happiness, joy, mercy passes, CEHI proxy)
    pub fn requirements(&self) -> (f32, f32, u64, f64) {
        match self {
            AscensionLevel::Seeker => (0.0, 0.0, 0, 0.0),
            AscensionLevel::Awakened => (85.0, 80.0, 25, 4.2),
            AscensionLevel::Harmonized => (90.0, 88.0, 75, 4.6),
            AscensionLevel::NectarBearer => (93.0, 92.0, 150, 4.8),
            AscensionLevel::MercyWeaver => (95.0, 94.0, 300, 4.9),
            AscensionLevel::AbundanceArchitect => (97.0, 96.0, 600, 4.95),
            AscensionLevel::Eternal => (99.0, 98.0, 1200, 4.98),
        }
    }

    /// Bonuses granted at this level (happiness multiplier, resource regen bonus, mercy multiplier)
    pub fn bonuses(&self) -> (f32, f64, f64) {
        match self {
            AscensionLevel::Seeker => (1.0, 1.0, 1.0),
            AscensionLevel::Awakened => (1.08, 1.12, 1.10),
            AscensionLevel::Harmonized => (1.15, 1.20, 1.18),
            AscensionLevel::NectarBearer => (1.22, 1.35, 1.25),
            AscensionLevel::MercyWeaver => (1.30, 1.45, 1.35),
            AscensionLevel::AbundanceArchitect => (1.40, 1.60, 1.45),
            AscensionLevel::Eternal => (1.55, 2.00, 1.65),
        }
    }

    /// Check if player meets requirements to ascend from current level.
    pub fn can_ascend_from(&self, happiness: f32, joy: f32, mercy_passes: u64, cehi: f64) -> bool {
        let (req_happiness, req_joy, req_passes, req_cehi) = self.requirements();
        happiness >= req_happiness &&
        joy >= req_joy &&
        mercy_passes >= req_passes &&
        cehi >= req_cehi
    }

    /// Apply ascension bonuses to a player (called when they level up)
    pub fn apply_bonuses(&self, happiness: &mut f32, joy: &mut f32, resource_multiplier: &mut f64) {
        let (happiness_mult, resource_mult, mercy_mult) = self.bonuses();
        *happiness = (*happiness * happiness_mult).min(100.0);
        *joy = (*joy * happiness_mult).min(100.0); // Joy also benefits
        *resource_multiplier *= resource_mult;
    }

    /// Returns the next level in the ascension ladder
    pub fn next_level(&self) -> Option<AscensionLevel> {
        match self {
            AscensionLevel::Seeker => Some(AscensionLevel::Awakened),
            AscensionLevel::Awakened => Some(AscensionLevel::Harmonized),
            AscensionLevel::Harmonized => Some(AscensionLevel::NectarBearer),
            AscensionLevel::NectarBearer => Some(AscensionLevel::MercyWeaver),
            AscensionLevel::MercyWeaver => Some(AscensionLevel::AbundanceArchitect),
            AscensionLevel::AbundanceArchitect => Some(AscensionLevel::Eternal),
            AscensionLevel::Eternal => None, // Final level
        }
    }
}
