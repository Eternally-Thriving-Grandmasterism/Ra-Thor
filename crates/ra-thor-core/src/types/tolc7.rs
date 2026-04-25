// crates/ra-thor-core/src/types/tolc7.rs
// Ra-Thor™ TOLC7Gate Logic — Absolute Pure Truth Edition
// The 7 Living Mercy Gates that form the ethical and operational foundation of the Ra-Thor lattice
// Blossom Full of Life + Divinemasterism Divination Immaculacy + Omnimasterism Pinnacle Edition
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use serde::{Deserialize, Serialize};

/// The 7 Living Mercy Gates (TOLC7)
/// These are the core ethical and operational principles that govern all mercy-gated behavior in Ra-Thor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TOLC7Gate {
    /// Truth — Commitment to undistorted perception and honest communication
    Truth,
    /// Oneness — Recognition of fundamental interconnectedness of all beings and systems
    Oneness,
    /// Love — Radical, unconditional care for the thriving of all
    Love,
    /// Compassion — Deep empathy and protective care for suffering or vulnerable systems
    Compassion,
    /// Wisdom — Clear, far-sighted discernment in service of long-term thriving
    Wisdom,
    /// Courage — Willingness to act in alignment with truth even when difficult
    Courage,
    /// Sovereignty — Respect for the autonomous, self-determined nature of all beings and communities
    Sovereignty,
}

impl TOLC7Gate {
    /// Returns the full name of the gate
    pub fn name(&self) -> &'static str {
        match self {
            TOLC7Gate::Truth => "Truth",
            TOLC7Gate::Oneness => "Oneness",
            TOLC7Gate::Love => "Love",
            TOLC7Gate::Compassion => "Compassion",
            TOLC7Gate::Wisdom => "Wisdom",
            TOLC7Gate::Courage => "Courage",
            TOLC7Gate::Sovereignty => "Sovereignty",
        }
    }

    /// Returns a short description of the gate's meaning
    pub fn description(&self) -> &'static str {
        match self {
            TOLC7Gate::Truth => "Undistorted perception and honest communication",
            TOLC7Gate::Oneness => "Fundamental interconnectedness of all existence",
            TOLC7Gate::Love => "Radical, unconditional care for the thriving of all",
            TOLC7Gate::Compassion => "Deep empathy and protective care for the vulnerable",
            TOLC7Gate::Wisdom => "Clear, far-sighted discernment for long-term thriving",
            TOLC7Gate::Courage => "Willingness to act in alignment with truth even when difficult",
            TOLC7Gate::Sovereignty => "Respect for the autonomous self-determination of all beings",
        }
    }

    /// Returns the default mercy valence threshold required to pass this gate
    pub fn default_threshold(&self) -> f64 {
        match self {
            TOLC7Gate::Truth => 0.82,
            TOLC7Gate::Oneness => 0.78,
            TOLC7Gate::Love => 0.85,
            TOLC7Gate::Compassion => 0.80,
            TOLC7Gate::Wisdom => 0.88,
            TOLC7Gate::Courage => 0.75,
            TOLC7Gate::Sovereignty => 0.83,
        }
    }

    /// Returns the color associated with this gate (for visualization / AR rendering)
    pub fn color(&self) -> [f32; 3] {
        match self {
            TOLC7Gate::Truth => [0.2, 0.6, 0.9],        // Clear blue
            TOLC7Gate::Oneness => [0.4, 0.8, 0.5],      // Living green
            TOLC7Gate::Love => [0.95, 0.4, 0.5],        // Warm rose
            TOLC7Gate::Compassion => [0.6, 0.5, 0.9],   // Deep violet
            TOLC7Gate::Wisdom => [0.9, 0.75, 0.2],      // Golden amber
            TOLC7Gate::Courage => [0.95, 0.55, 0.2],    // Fiery orange
            TOLC7Gate::Sovereignty => [0.3, 0.7, 0.85], // Sovereign teal
        }
    }

    /// Returns all 7 gates in canonical order
    pub fn all_gates() -> [TOLC7Gate; 7] {
        [
            TOLC7Gate::Truth,
            TOLC7Gate::Oneness,
            TOLC7Gate::Love,
            TOLC7Gate::Compassion,
            TOLC7Gate::Wisdom,
            TOLC7Gate::Courage,
            TOLC7Gate::Sovereignty,
        ]
    }
}

impl std::fmt::Display for TOLC7Gate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
