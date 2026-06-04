//! crates/powrush/src/device_capability.rs
//! Device Capability Detection for Powrush-MMO
//! Enables adaptive experiences across PC, Mobile, Web, and future devices.

use bevy::prelude::*;

/// Represents the type of device/platform the game is running on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Platform {
    #[default]
    Desktop,
    Mobile,
    Web,
    Console,
    Unknown,
}

/// Represents input capabilities of the current device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputMethod {
    #[default]
    KeyboardMouse,
    Touch,
    Gamepad,
    Mixed,
}

/// Resource that describes the current device's capabilities.
/// Used to drive adaptive UI and feature availability.
#[derive(Resource, Debug, Clone)]
pub struct DeviceCapability {
    pub platform: Platform,
    pub input_method: InputMethod,
    pub screen_width: f32,
    pub screen_height: f32,
    pub supports_high_end_graphics: bool,
}

impl Default for DeviceCapability {
    fn default() -> Self {
        Self {
            platform: Platform::Desktop,
            input_method: InputMethod::KeyboardMouse,
            screen_width: 1920.0,
            screen_height: 1080.0,
            supports_high_end_graphics: true,
        }
    }
}

impl DeviceCapability {
    /// Creates a new DeviceCapability with detected values.
    /// In a real implementation, this would use platform-specific detection.
    pub fn detect() -> Self {
        // TODO: Replace with actual platform detection (wasm, mobile, etc.)
        // For now, we default to a capable Desktop environment.
        Self::default()
    }

    /// Returns true if this device should use the simplified MobileTown experience.
    pub fn is_mobile_like(&self) -> bool {
        matches!(self.platform, Platform::Mobile | Platform::Web) ||
        (self.screen_width < 900.0 || self.screen_height < 700.0)
    }
}

/// Plugin that inserts the DeviceCapability resource.
pub struct DeviceCapabilityPlugin;

impl Plugin for DeviceCapabilityPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DeviceCapability>();
        app.add_systems(Startup, detect_device_capabilities);
    }
}

fn detect_device_capabilities(mut commands: Commands) {
    let capability = DeviceCapability::detect();
    commands.insert_resource(capability);
    info!("[Powrush] DeviceCapability detected: {:?}", capability);
}
