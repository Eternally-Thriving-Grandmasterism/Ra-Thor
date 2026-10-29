//! crates/powrush/src/device_capability.rs
//! Device Capability Detection for Powrush-MMO

use bevy::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Platform {
    #[default]
    Desktop,
    Mobile,
    Web,
    Console,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputMethod {
    #[default]
    KeyboardMouse,
    Touch,
    Gamepad,
    Mixed,
}

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
    /// Detects platform using compile-time cfg attributes.
    pub fn detect() -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            return Self {
                platform: Platform::Web,
                input_method: InputMethod::Touch,
                screen_width: 800.0,
                screen_height: 600.0,
                supports_high_end_graphics: false,
            };
        }

        #[cfg(any(target_os = "android", target_os = "ios"))]
        {
            return Self {
                platform: Platform::Mobile,
                input_method: InputMethod::Touch,
                screen_width: 400.0,
                screen_height: 800.0,
                supports_high_end_graphics: false,
            };
        }

        Self::default()
    }

    pub fn is_mobile_like(&self) -> bool {
        matches!(self.platform, Platform::Mobile | Platform::Web) ||
            (self.screen_width < 900.0 || self.screen_height < 700.0)
    }
}

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
