//! # Faction System (v0.1.0)
//!
//! The seven living factions of Powrush — each deeply aligned with
//! specific 7 Living Mercy Gates and RBE principles.
//! Diplomacy, alliances, mercy compliance, and ascension bonuses
//! are all governed here.

use crate::mercy::MercyGateStatus;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Faction {
    Ambrosians,              // Joy Amplification + Ambrosian Nectar
    Harmonists,              // Harmony Preservation
    Truthseekers,            // Truth Verification
    AbundanceBuilders,       // Abundance Creation
    MercyWeavers,            // Ethical Alignment + NonDeception
    PostScarcityEngineers,   // Post-Scarcity Enforcement
    EternalCompassion,       // The heart of TOLC — all gates amplified
}

impl Faction {
    pub fn name(&self) -> &'static str {
        match self {
            Faction::Ambrosians => "Ambrosians",
            Faction::Harmonists => "Harmonists",
            Faction::Truthseekers => "Truthseekers",
            Faction::AbundanceBuilders => "Abundance Builders",
            Faction::MercyWeavers => "Mercy Weavers",
            Faction::PostScarcityEngineers => "Post-Scarcity Engineers",
            Faction::EternalCompassion => "Eternal Compassion",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Faction::Ambrosians => "Masters of collective joy, nectar, and bliss. Strongest connection to Joy Amplification gate.",
            Faction::Harmonists => "Preservers of perfect balance and peace. Masters of Harmony Preservation.",
            Faction::Truthseekers => "Guardians of absolute truth. Never compromise on Truth Verification.",
            Faction::AbundanceBuilders => "Creators of infinite resources for all beings. Aligned with Abundance Creation.",
            Faction::MercyWeavers => "Weave mercy into every decision. Strongest Ethical Alignment + NonDeception.",
            Faction::PostScarcityEngineers => "Engineers of a world where scarcity no longer exists. Post-Scarcity Enforcement.",
            Faction::EternalCompassion => "The living heart of TOLC. Amplifies all 7 Mercy Gates simultaneously.",
        }
    }

    /// Mercy multiplier bonus for this faction (higher = stronger mercy alignment)
    pub fn mercy_bonus(&self) -> f64 {
        match self {
            Faction::Ambrosians => 1.25,
            Faction::Harmonists => 1.20,
            Faction::Truthseekers => 1.15,
            Faction::AbundanceBuilders => 1.30,
            Faction::MercyWeavers => 1.40,
            Faction::PostScarcityEngineers => 1.35,
            Faction::EternalCompassion => 1.55,
        }
    }

    /// Check if a proposed action complies with this faction's core mercy principles.
    pub fn check_mercy_compliance(&self, action_description: &str) -> MercyGateStatus {
        let action = action_description.to_lowercase();

        match self {
            Faction::Truthseekers if action.contains("lie") || action.contains("deceive") || action.contains("mislead") => {
                MercyGateStatus::Failed
            }
            Faction::Harmonists if action.contains("conflict") || action.contains("war") || action.contains("divide") => {
                MercyGateStatus::Failed
            }
            Faction::AbundanceBuilders if action.contains("hoard") || action.contains("scarcity") || action.contains("monopoly") => {
                MercyGateStatus::Failed
            }
            Faction::MercyWeavers if action.contains("harm") || action.contains("exploit") || action.contains("manipulate") => {
                MercyGateStatus::Failed
            }
            Faction::PostScarcityEngineers if action.contains("limit") || action.contains("restrict") || action.contains("gatekeep") => {
                MercyGateStatus::Failed
            }
            _ => MercyGateStatus::Passed,
        }
    }

    /// Diplomacy bonus between two factions (higher = better relations)
    pub fn get_diplomacy_bonus(&self, other: Faction) -> f64 {
        match (self, other) {
            (Faction::Ambrosians, Faction::EternalCompassion) | (Faction::EternalCompassion, Faction::Ambrosians) => 1.35,
            (Faction::Harmonists, Faction::MercyWeavers) | (Faction::MercyWeavers, Faction::Harmonists) => 1.30,
            (Faction::Truthseekers, Faction::EternalCompassion) | (Faction::EternalCompassion, Faction::Truthseekers) => 1.25,
            (Faction::AbundanceBuilders, Faction::PostScarcityEngineers) | (Faction::PostScarcityEngineers, Faction::AbundanceBuilders) => 1.28,
            (Faction::Ambrosians, Faction::Harmonists) | (Faction::Harmonists, Faction::Ambrosians) => 1.15,
            _ => 1.05, // Neutral positive baseline (RBE encourages cooperation)
        }
    }

    /// Returns all seven factions as a vector (useful for UI and simulation)
    pub fn all_factions() -> Vec<Faction> {
        vec![
            Faction::Ambrosians,
            Faction::Harmonists,
            Faction::Truthseekers,
            Faction::AbundanceBuilders,
            Faction::MercyWeavers,
            Faction::PostScarcityEngineers,
            Faction::EternalCompassion,
        ]
    }
}
