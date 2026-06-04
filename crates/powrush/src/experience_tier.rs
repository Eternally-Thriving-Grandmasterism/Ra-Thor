//! crates/powrush/src/experience_tier.rs
//! Experience Tier system for Powrush-MMO
//! Controls feature availability based on device capability.

use bevy::prelude::*;
use crate::device_capability::DeviceCapability;

/// Defines the experience tier the player is currently in.
/// This drives UI complexity and available gameplay systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Resource, Default)]
pub enum ExperienceTier {
    /// Full desktop experience with outdoor combat, deep systems, and intricate UI.
    #[default]
    DesktopFull,
    /// Simplified mobile/town-focused experience.
    /// No outdoor combat. Full access to town activities, socializing, crafting, banking, and AGI NPCs.
    MobileTown,
    /// Future: Balanced experience for tablets / mid-range devices.
    TabletBalanced,
}

impl ExperienceTier {
    /// Automatically determines the best tier based on DeviceCapability.
    pub fn from_device_capability(capability: &DeviceCapability) -> Self {
        if capability.is_mobile_like() {
            Self::MobileTown
        } else {
            Self::DesktopFull
        }
    }

    /// Returns true if outdoor combat and advanced world exploration should be available.
    pub fn allows_outdoor_combat(&self) -> bool {
        matches!(self, Self::DesktopFull | Self::TabletBalanced)
    }

    /// Returns true if the full intricate UI should be shown.
    pub fn has_advanced_ui(&self) -> bool {
        matches!(self, Self::DesktopFull)
    }

    /// Core Town Loop features that should always be available (even on mobile).
    pub fn has_core_town_loop(&self) -> bool {
        true // Always available across all tiers
    }
}

/// Plugin that manages ExperienceTier based on DeviceCapability.
pub struct ExperienceTierPlugin;

impl Plugin for ExperienceTierPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ExperienceTier>();
        app.add_systems(Update, update_experience_tier);
    }
}

fn update_experience_tier(
    device: Res<DeviceCapability>,
    mut tier: ResMut<ExperienceTier>,
) {
    let new_tier = ExperienceTier::from_device_capability(&device);

    if *tier != new_tier {
        *tier = new_tier;
        info!("[Powrush] ExperienceTier changed to: {:?}", new_tier);
    }
}
