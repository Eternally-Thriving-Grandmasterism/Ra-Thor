//! crates/powrush/src/virtual_joystick.rs
//! Virtual Joystick for Powrush-MMO Mobile Experience
//! PATSAGi + Ra-Thor approved: Client-Side Prediction + Reconciliation foundation
//! Eternal even playing field + maximal end-user joy, responsiveness, positive emotion

use bevy::prelude::*;
use bevy_egui::EguiContexts;
use egui::{Color32, Sense, Vec2 as EguiVec2};
use crate::experience_tier::ExperienceTier;
use crate::simulation::Position; // shared deterministic type

/// Bevy Event carrying joystick input for decoupled CouncilProposal generation.
/// Sequence number enables Server Reconciliation (rewind + replay) for eternal fair play.
#[derive(Event, Debug, Clone, Copy, Default)]
pub struct JoystickMoveEvent {
    pub sequence: u64,
    pub dx: f32,
    pub dy: f32,
    pub is_active: bool,
    pub intensity: f32,
}

/// Resource holding current joystick state + prediction buffer for reconciliation.
#[derive(Resource, Debug, Default)]
pub struct VirtualJoystick {
    pub movement: Vec2,
    pub is_active: bool,
    pub center: EguiVec2,
    pub current_pos: EguiVec2,
    pub sensitivity: f32,
    pub deadzone: f32,
    /// Input history buffer for Server Reconciliation (seq + event). Eternal fair play.
    pub pending_inputs: Vec<JoystickMoveEvent>,
    next_sequence: u64,
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
            pending_inputs: Vec::new(),
            next_sequence: 1,
        }
    }

    pub fn direction(&self) -> Vec2 {
        if self.movement.length() < self.deadzone { Vec2::ZERO } else { self.movement * self.sensitivity }
    }

    /// Record input for later reconciliation (called after event send).
    pub fn record_pending(&mut self, event: JoystickMoveEvent) {
        self.pending_inputs.push(event);
        if self.pending_inputs.len() > 64 { self.pending_inputs.remove(0); } // bounded history
    }

    pub fn next_sequence(&mut self) -> u64 {
        let seq = self.next_sequence;
        self.next_sequence += 1;
        seq
    }
}

/// Plugin with prediction-ready systems.
pub struct VirtualJoystickPlugin;

impl Plugin for VirtualJoystickPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VirtualJoystick>();
        app.add_event::<JoystickMoveEvent>();
        app.add_systems(Update, (
            show_virtual_joystick.run_if(|tier: Res<ExperienceTier>| *tier == ExperienceTier::MobileTown),
            reset_joystick_when_inactive,
            emit_joystick_events,
            apply_local_prediction, // immediate responsive joy (CSP foundation)
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

            ui.painter().circle_filled(center, size / 2.0, Color32::from_gray(55));

            let knob_radius = size / 4.0;
            let knob_pos = if response.dragged() {
                let delta = response.drag_delta();
                let new_pos = joystick.current_pos + delta;
                let max_distance = size / 2.0 - knob_radius;
                let dir = (new_pos - center).normalized();
                let distance = (new_pos - center).length().min(max_distance);
                center + dir * distance
            } else { center };

            ui.painter().circle_filled(knob_pos, knob_radius, Color32::from_gray(210));

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

/// Emits sequenced event + records for reconciliation. Decoupled for CouncilProposal bridge.
fn emit_joystick_events(
    mut joystick: ResMut<VirtualJoystick>,
    mut events: EventWriter<JoystickMoveEvent>,
) {
    if joystick.is_active || joystick.movement != Vec2::ZERO {
        let seq = joystick.next_sequence();
        let intensity = joystick.movement.length().clamp(0.0, 1.0);
        let event = JoystickMoveEvent {
            sequence: seq,
            dx: joystick.movement.x,
            dy: joystick.movement.y,
            is_active: joystick.is_active,
            intensity,
        };
        events.send(event);
        joystick.record_pending(event);
    }
}

/// Local Client-Side Prediction hook (approved CSP foundation).
/// Immediately applies movement for joyful responsive feel.
/// Uses identical deterministic logic as authoritative simulation (see predict_move_position).
/// Does NOT bypass council truth — reconciliation will correct when authoritative snapshot arrives.
fn apply_local_prediction(
    mut joystick: Res<VirtualJoystick>,
    // In real integration: mut query for local player visual/entity
) {
    // Placeholder: in full client, call simulation::predict_move_position on local predicted state here.
    // Example future:
    // if let Some(mut pos) = local_player_position {
    //     *pos = crate::simulation::predict_move_position(*pos, joystick.movement.x, joystick.movement.y);
    // }
    let _ = joystick; // keep resource alive for buffer
}
