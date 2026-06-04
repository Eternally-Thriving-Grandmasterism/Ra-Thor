//! crates/powrush/src/virtual_joystick.rs
//! Virtual Joystick for Powrush-MMO Mobile Experience
//! Event-driven foundation for CouncilProposal routing (option 2 locked)
//! MMO priority: maximal user joy, responsive feel, future presets + granular custom controls

use bevy::prelude::*;
use bevy_egui::EguiContexts;
use egui::{pos2, Color32, Sense, Ui, Vec2 as EguiVec2};
use crate::experience_tier::ExperienceTier;

/// Lightweight Bevy Event for decoupled input (C layer).
/// Listeners (bridge systems) turn this into CouncilProposal { IssueCommand(MovePlayer), mercy_evaluation, ... }
/// and route via ShardManager::route_proposal for manifold + epigenetic blessings.
/// Keeps UI snappy and joyful — proposal routing happens without blocking drag feel.
#[derive(Event, Debug, Clone, Copy, Default)]
pub struct JoystickMoveEvent {
    pub dx: f32,
    pub dy: f32,
    pub is_active: bool,
    /// Future: sensitivity, layout preset id, custom deadzone, etc. for granular player controls
    pub intensity: f32,
}

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
    /// Extensibility for custom settings / presets (MMO joy & comfort)
    pub sensitivity: f32,
    pub deadzone: f32,
}

impl VirtualJoystick {
    pub fn new() -> Self {
        Self {
            movement: Vec2::ZERO,
            is_active: false,
            center: EguiVec2::new(0.0, 0.0),
            current_pos: EguiVec2::new(0.0, 0.0),
            sensitivity: 1.0,
            deadzone: 0.1,
        }
    }

    /// Returns the movement direction as a normalized Vec2 (clamped by deadzone).
    pub fn direction(&self) -> Vec2 {
        if self.movement.length() < self.deadzone {
            Vec2::ZERO
        } else {
            self.movement * self.sensitivity
        }
    }
}

/// Plugin that adds the virtual joystick + event.
pub struct VirtualJoystickPlugin;

impl Plugin for VirtualJoystickPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VirtualJoystick>();
        app.add_event::<JoystickMoveEvent>();
        app.add_systems(Update, (
            show_virtual_joystick.run_if(|tier: Res<ExperienceTier>| *tier == ExperienceTier::MobileTown),
            reset_joystick_when_inactive,
            emit_joystick_events,
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

            // Draw outer circle (background) — warm, inviting for joy
            ui.painter().circle_filled(center, size / 2.0, Color32::from_gray(55));

            // Draw inner circle (knob) — responsive, satisfying tactile feel
            let knob_radius = size / 4.0;
            let knob_pos = if response.dragged() {
                let delta = response.drag_delta();
                let new_pos = joystick.current_pos + delta;
                let max_distance = size / 2.0 - knob_radius;

                let dir = (new_pos - center).normalized();
                let distance = (new_pos - center).length().min(max_distance);
                center + dir * distance
            } else {
                center
            };

            ui.painter().circle_filled(knob_pos, knob_radius, Color32::from_gray(210));

            // Update joystick state
            if response.dragged() {
                joystick.is_active = true;
                joystick.current_pos = knob_pos;
                joystick.center = center;

                let delta = knob_pos - center;
                let raw_x = (delta.x / (size / 2.0)).clamp(-1.0, 1.0);
                let raw_y = (delta.y / (size / 2.0)).clamp(-1.0, 1.0);
                joystick.movement = Vec2::new(raw_x, raw_y);
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

/// Emits JoystickMoveEvent for decoupled listeners (bridge to CouncilProposal).
/// This keeps the UI layer pure and joyful — proposal generation + manifold routing happens elsewhere.
fn emit_joystick_events(
    joystick: Res<VirtualJoystick>,
    mut events: EventWriter<JoystickMoveEvent>,
) {
    if joystick.is_active || joystick.movement != Vec2::ZERO {
        let intensity = joystick.movement.length().clamp(0.0, 1.0);
        events.send(JoystickMoveEvent {
            dx: joystick.movement.x,
            dy: joystick.movement.y,
            is_active: joystick.is_active,
            intensity,
        });
    }
}
