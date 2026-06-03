use bevy::prelude::*;

/// Tracks which Ra-Thor / PATSAGi-powered systems the server has unlocked
/// and their current activation status for the weekly war.
#[derive(Resource, Default)]
pub struct ServerUnlockState {
    /// Tier 1
    pub epigenetic_surge_unlocked: bool,
    pub geometric_beacon_unlocked: bool,

    /// Tier 2
    pub council_oversight_unlocked: bool,
    pub layer_shift_beacon_unlocked: bool,

    /// Tier 3 (Advanced)
    pub ra_thor_tactical_lattice_unlocked: bool,
    pub agi_rbe_field_projection_unlocked: bool,

    /// Current activation status during war
    pub epigenetic_surge_active: bool,
    pub geometric_beacon_active: bool,
    pub council_oversight_active: bool,
    pub layer_shift_beacon_active: bool,
    pub ra_thor_tactical_lattice_active: bool,
    pub agi_rbe_field_projection_active: bool,

    /// Progress toward next unlocks (0.0 - 1.0)
    pub council_influence_progress: f32,
}

impl ServerUnlockState {
    pub fn can_activate_tier_1(&self) -> bool {
        self.epigenetic_surge_unlocked || self.geometric_beacon_unlocked
    }

    pub fn can_activate_tier_2(&self) -> bool {
        self.council_oversight_unlocked || self.layer_shift_beacon_unlocked
    }

    pub fn can_activate_tier_3(&self) -> bool {
        self.ra_thor_tactical_lattice_unlocked || self.agi_rbe_field_projection_unlocked
    }
}