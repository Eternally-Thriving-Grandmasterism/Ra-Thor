//! crates/powrush/src/virtual_joystick.rs
//! Virtual Joystick for Powrush-MMO Mobile Experience

use bevy::prelude::*;
use bevy_egui::EguiContexts;
use egui::{pos2, Color32, Sense, Ui, Vec2 as EguiVec2};
use crate::experience_tier::ExperienceTier;

/// Resource holding the current virtual joystick state.
#[derive(Resource, Debug, Default)]
pub struct VirtualJoystick {
    /// Normalized movement vector (-1.0 to 1.0).
    pub movement: Vec2,
    /// Whether the joystick is currently being touched/dragged.
    pub is_active: bool,
    /// Position of the joystick center (in egui coordinates).
    pub center: EguiVec2,
    /// Current touch/drag position.
    pub current_pos: EguiVec2,
}

impl VirtualJoystick {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the movement direction as a normalized Vec2.
    pub fn direction(&self) -> Vec2 {
        self.movement
    }
}

/// Plugin that adds the virtual joystick.
pub struct VirtualJoystickPlugin;

impl Plugin for VirtualJoystickPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VirtualJoystick>();
        app.add_systems(Update, (
            show_virtual_joystick.run_if(|tier: Res<ExperienceTier>| *tier == ExperienceTier::MobileTown),
            reset_joystick_when_inactive,
        ));
    }
}

fn show_virtual_joystick(
    mut contexts: EguiContexts,
    mut joystick: ResMut<VirtualJoystick>,
) {
    let ctx = contexts.ctx_mut();

    egui::Area::new("virtual_joystick")
        .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(60.0, -120.0))
        .show(ctx, |ui| {
            let size = 120.0;
            let response = ui.allocate_response(
                egui::vec2(size, size),
                Sense::drag(),
            );

            let rect = response.rect;
            let center = rect.center();

            // Draw outer circle (background)
            ui.painter().circle_filled(center, size / 2.0, Color32::from_gray(60));

            // Draw inner circle (knob)
            let knob_radius = size / 4.0;
            let knob_pos = if response.dragged() {
                let delta = response.drag_delta();
                let new_pos = joystick.current_pos + delta;
                let max_distance = size / 2.0 - knob_radius;

                // Clamp to circle
                let dir = (new_pos - center).normalized();
                let distance = (new_pos - center).length().min(max_distance);
                center + dir * distance
            } else {
                center
            };

            ui.painter().circle_filled(knob_pos, knob_radius, Color32::from_gray(200));

            // Update joystick state
            if response.dragged() {
                joystick.is_active = true;
                joystick.current_pos = knob_pos;
                joystick.center = center;

                let delta = knob_pos - center;
                joystick.movement = Vec2::new(
                    (delta.x / (size / 2.0)).clamp(-1.0, 1.0),
                    (delta.y / (size / 2.0)).clamp(-1.0, 1.0),
                );
            } else {
                joystick.is_active = false;
            }
        });
}

fn reset_joystick_when_inactive(mut joystick: ResMut<VirtualJoystick>) {
    if !joystick.is_active {
        joystick.movement = Vec2::ZERO;
        joystick.current_pos = joystick.center;
    }
}
