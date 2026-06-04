//! crates/powrush/src/virtual_joystick.rs
//! Virtual Joystick for Powrush-MMO Mobile Experience

use bevy::prelude::*;
use bevy_egui::EguiContexts;
use egui::{Color32, Sense};
use crate::experience_tier::ExperienceTier;
use crate::simulation::{WorldSimulation, SimulationCommand};

#[derive(Component, Debug, Default)]
pub struct Player;

#[derive(Resource, Debug, Default)]
pub struct VirtualJoystick {
    pub movement: Vec2,
    pub is_active: bool,
    pub center: egui::Vec2,
    pub current_pos: egui::Vec2,
}

impl VirtualJoystick {
    pub fn direction(&self) -> Vec2 {
        self.movement
    }
}

pub struct VirtualJoystickPlugin;

impl Plugin for VirtualJoystickPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VirtualJoystick>();
        app.add_systems(Update, (
            show_virtual_joystick.run_if(|tier: Res<ExperienceTier>| *tier == ExperienceTier::MobileTown),
            apply_joystick_to_simulation.run_if(|tier: Res<ExperienceTier>| *tier == ExperienceTier::MobileTown),
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
            let response = ui.allocate_response(egui::vec2(size, size), Sense::drag());
            let rect = response.rect;
            let center = rect.center();

            ui.painter().circle_filled(center, size / 2.0, Color32::from_gray(60));
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

            ui.painter().circle_filled(knob_pos, knob_radius, Color32::from_gray(200));

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

fn apply_joystick_to_simulation(
    joystick: Res<VirtualJoystick>,
    mut simulation: ResMut<WorldSimulation>,
    time: Res<Time>,
) {
    if !joystick.is_active || joystick.movement == Vec2::ZERO {
        return;
    }

    let speed = 300.0;
    let dt = time.delta_seconds();
    let dx = joystick.movement.x * speed * dt;
    let dy = joystick.movement.y * speed * dt;

    // Use the existing SimulationCommand pattern for authoritative simulation
    let command = SimulationCommand::MovePlayer { dx, dy };
    simulation.evaluate_command_with_mercy(&command, None);

    // Update internal player position for immediate feedback
    simulation.player.position.x += dx;
    simulation.player.position.y += dy;
}

fn reset_joystick_when_inactive(mut joystick: ResMut<VirtualJoystick>) {
    if !joystick.is_active {
        joystick.movement = Vec2::ZERO;
        joystick.current_pos = joystick.center;
    }
}
