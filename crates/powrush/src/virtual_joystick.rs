//! crates/powrush/src/virtual_joystick.rs
//! Virtual Joystick for Powrush-MMO Mobile Experience
//! Rollback Netcode style Server Reconciliation (option 2 locked)
//! Higher fidelity rollback + replay for eternal even playing field + maximal joy

use bevy::prelude::*;
use bevy_egui::EguiContexts;
use egui::{Color32, Sense, Vec2 as EguiVec2};
use crate::experience_tier::ExperienceTier;
use crate::simulation::{Position, predict_move_position};

/// Bevy Event with sequence for rollback reconciliation.
#[derive(Event, Debug, Clone, Copy, Default)]
pub struct JoystickMoveEvent {
    pub sequence: u64,
    pub dx: f32,
    pub dy: f32,
    pub is_active: bool,
    pub intensity: f32,
}

/// Snapshot of predicted state at a specific sequence (for rollback).
#[derive(Debug, Clone, Copy)]
pub struct PredictedState {
    pub sequence: u64,
    pub position: Position,
    pub harmony: f64, // included for future mercy-aware rollback
}

/// Resource with input buffer + rollback history.
#[derive(Resource, Debug, Default)]
pub struct VirtualJoystick {
    pub movement: Vec2,
    pub is_active: bool,
    pub center: EguiVec2,
    pub current_pos: EguiVec2,
    pub sensitivity: f32,
    pub deadzone: f32,
    pub pending_inputs: Vec<JoystickMoveEvent>,
    next_sequence: u64,
    /// Rollback history: past predicted states (ring buffer for efficiency).
    pub prediction_history: Vec<PredictedState>,
    pub current_predicted_position: Position,
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
            prediction_history: Vec::new(),
            current_predicted_position: Position { x: 0.0, y: 0.0 },
        }
    }

    pub fn direction(&self) -> Vec2 {
        if self.movement.length() < self.deadzone { Vec2::ZERO } else { self.movement * self.sensitivity }
    }

    pub fn record_pending(&mut self, event: JoystickMoveEvent) {
        self.pending_inputs.push(event);
        if self.pending_inputs.len() > 128 { self.pending_inputs.remove(0); }
    }

    pub fn next_sequence(&mut self) -> u64 {
        let seq = self.next_sequence;
        self.next_sequence += 1;
        seq
    }

    /// Record current predicted state for future rollback (called after local prediction).
    pub fn record_prediction_state(&mut self, state: PredictedState) {
        self.prediction_history.push(state);
        if self.prediction_history.len() > 128 { self.prediction_history.remove(0); }
        self.current_predicted_position = state.position;
    }

    /// Rollback to a specific sequence and return the state (core of Rollback Netcode).
    /// Then caller replays all newer pending inputs.
    pub fn rollback_to_sequence(&mut self, target_seq: u64) -> Option<PredictedState> {
        if let Some(pos) = self.prediction_history.iter().rposition(|s| s.sequence <= target_seq) {
            let state = self.prediction_history[pos];
            self.prediction_history.truncate(pos + 1);
            self.current_predicted_position = state.position;
            // Remove pending inputs before or at target
            self.pending_inputs.retain(|e| e.sequence > target_seq);
            Some(state)
        } else {
            None
        }
    }

    /// Replay a list of events after rollback using the pure deterministic predictor.
    pub fn replay_events(&mut self, events: &[JoystickMoveEvent]) {
        for event in events {
            self.current_predicted_position = predict_move_position(
                self.current_predicted_position,
                event.dx,
                event.dy,
            );
            // In full integration, also re-emit CouncilProposal here if needed
        }
    }
}

/// Plugin
pub struct VirtualJoystickPlugin;

impl Plugin for VirtualJoystickPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VirtualJoystick>();
        app.add_event::<JoystickMoveEvent>();
        app.add_systems(Update, (
            show_virtual_joystick.run_if(|tier: Res<ExperienceTier>| *tier == ExperienceTier::MobileTown),
            reset_joystick_when_inactive,
            emit_joystick_events,
            apply_local_prediction,
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

        // Record predicted state for rollback history
        let new_pos = predict_move_position(joystick.current_predicted_position, event.dx, event.dy);
        joystick.record_prediction_state(PredictedState {
            sequence: seq,
            position: new_pos,
            harmony: 0.75, // placeholder; real value from local simulation
        });
    }
}

/// Local CSP with rollback support hook.
fn apply_local_prediction(mut joystick: ResMut<VirtualJoystick>) {
    // Future: when authoritative correction arrives, call:
    // if let Some(state) = joystick.rollback_to_sequence(correction_seq) {
    //     joystick.replay_events(&joystick.pending_inputs);
    // }
    let _ = joystick;
}
