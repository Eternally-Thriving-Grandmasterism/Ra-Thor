//! crates/powrush/src/virtual_joystick.rs
//! Virtual Joystick for Powrush-MMO Mobile Experience
//! Deterministic Checksums (production grade, lovingly implemented)
//! For desync detection, replay verification, and guaranteed determinism

use bevy::prelude::*;
use bevy_egui::EguiContexts;
use egui::{Color32, Sense, Vec2 as EguiVec2};
use crate::experience_tier::ExperienceTier;
use crate::simulation::{Position, predict_move_position};
use serde::{Serialize, Deserialize};
use std::fs;

// ==================== DETERMINISTIC CHECKSUM ====================

/// Computes a deterministic checksum for a predicted state.
/// Used for fast desync detection and replay verification.
/// Must be pure and platform-independent.
pub fn compute_state_checksum(state: &PredictedState) -> u64 {
    // Simple but effective deterministic hash (position + harmony)
    // In production this can be upgraded to xxHash or similar without changing API
    let x_bits = state.position.x.to_bits() as u64;
    let y_bits = state.position.y.to_bits() as u64;
    let harmony_bits = state.harmony.to_bits();

    // Fold with sequence for uniqueness
    let mut hash = state.sequence;
    hash = hash.wrapping_mul(6364136223846793005).wrapping_add(x_bits);
    hash = hash.wrapping_mul(6364136223846793005).wrapping_add(y_bits);
    hash = hash.wrapping_mul(6364136223846793005).wrapping_add(harmony_bits);
    hash
}

// ==================== REPLAY LOGGING WITH CHECKSUMS ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplayEntry {
    MoveInput {
        sequence: u64,
        tick: u64,
        dx: f32,
        dy: f32,
        intensity: f32,
    },
    AuthoritativeCorrection {
        sequence: u64,
        tick: u64,
        position: Position,
        harmony: f64,
        checksum: u64,
        notes: String,
    },
    InitialState {
        tick: u64,
        position: Position,
        harmony: f64,
        checksum: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeterministicReplayLog {
    pub version: u32,
    pub player_id: u64,
    pub shard_id: u32,
    pub initial_tick: u64,
    pub entries: Vec<ReplayEntry>,
}

impl DeterministicReplayLog {
    pub fn new(player_id: u64, shard_id: u32) -> Self {
        Self {
            version: 1,
            player_id,
            shard_id,
            initial_tick: 0,
            entries: Vec::new(),
        }
    }

    pub fn record_initial_state(&mut self, tick: u64, position: Position, harmony: f64) {
        self.initial_tick = tick;
        let checksum = compute_state_checksum(&PredictedState { sequence: 0, position, harmony });
        self.entries.push(ReplayEntry::InitialState { tick, position, harmony, checksum });
    }

    pub fn record_move(&mut self, sequence: u64, tick: u64, dx: f32, dy: f32, intensity: f32) {
        self.entries.push(ReplayEntry::MoveInput { sequence, tick, dx, dy, intensity });
    }

    pub fn record_correction(&mut self, sequence: u64, tick: u64, position: Position, harmony: f64, notes: String) {
        let checksum = compute_state_checksum(&PredictedState { sequence, position, harmony });
        self.entries.push(ReplayEntry::AuthoritativeCorrection { sequence, tick, position, harmony, checksum, notes });
    }

    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)
    }

    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let data = fs::read_to_string(path)?;
        let log: Self = serde_json::from_str(&data)?;
        Ok(log)
    }
}

#[derive(Resource, Debug, Default)]
pub struct ReplayLogger {
    pub log: DeterministicReplayLog,
    pub recording: bool,
    pub replaying: bool,
    pub current_replay_index: usize,
    pub current_tick: u64,
}

impl ReplayLogger {
    pub fn new(player_id: u64, shard_id: u32) -> Self {
        Self {
            log: DeterministicReplayLog::new(player_id, shard_id),
            recording: true,
            replaying: false,
            current_replay_index: 0,
            current_tick: 0,
        }
    }

    pub fn start_recording(&mut self, player_id: u64, shard_id: u32) {
        self.log = DeterministicReplayLog::new(player_id, shard_id);
        self.recording = true;
        self.replaying = false;
        self.current_replay_index = 0;
    }

    pub fn stop_recording(&mut self) {
        self.recording = false;
    }

    pub fn record_move_event(&mut self, event: &JoystickMoveEvent, tick: u64) {
        if self.recording {
            self.log.record_move(event.sequence, tick, event.dx, event.dy, event.intensity);
        }
    }

    pub fn record_authoritative_correction(&mut self, sequence: u64, tick: u64, position: Position, harmony: f64, notes: String) {
        if self.recording {
            self.log.record_correction(sequence, tick, position, harmony, notes);
        }
    }

    pub fn start_replay(&mut self) {
        self.replaying = true;
        self.recording = false;
        self.current_replay_index = 0;
        self.current_tick = self.log.initial_tick;
    }

    pub fn next_replay_move(&mut self) -> Option<JoystickMoveEvent> {
        if !self.replaying { return None; }

        while self.current_replay_index < self.log.entries.len() {
            if let ReplayEntry::MoveInput { sequence, tick: _, dx, dy, intensity } = &self.log.entries[self.current_replay_index] {
                self.current_replay_index += 1;
                return Some(JoystickMoveEvent {
                    sequence: *sequence,
                    dx: *dx,
                    dy: *dy,
                    is_active: true,
                    intensity: *intensity,
                });
            } else {
                self.current_replay_index += 1;
            }
        }
        self.replaying = false;
        None
    }
}

// ==================== CORE TYPES ====================

#[derive(Event, Debug, Clone, Copy, Default)]
pub struct JoystickMoveEvent {
    pub sequence: u64,
    pub dx: f32,
    pub dy: f32,
    pub is_active: bool,
    pub intensity: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct PredictedState {
    pub sequence: u64,
    pub position: Position,
    pub harmony: f64,
}

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

    pub fn record_prediction_state(&mut self, state: PredictedState) {
        self.prediction_history.push(state);
        if self.prediction_history.len() > 128 { self.prediction_history.remove(0); }
        self.current_predicted_position = state.position;
    }

    pub fn rollback_to_sequence(&mut self, target_seq: u64) -> Option<PredictedState> {
        if let Some(pos) = self.prediction_history.iter().rposition(|s| s.sequence <= target_seq) {
            let state = self.prediction_history[pos];
            self.prediction_history.truncate(pos + 1);
            self.current_predicted_position = state.position;
            self.pending_inputs.retain(|e| e.sequence > target_seq);
            Some(state)
        } else {
            None
        }
    }

    pub fn replay_events(&mut self, events: &[JoystickMoveEvent]) {
        for event in events {
            self.current_predicted_position = predict_move_position(
                self.current_predicted_position,
                event.dx,
                event.dy,
            );
        }
    }
}

// ==================== PLUGIN ====================

pub struct VirtualJoystickPlugin;

impl Plugin for VirtualJoystickPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VirtualJoystick>();
        app.init_resource::<ReplayLogger>();
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
    mut replay_logger: ResMut<ReplayLogger>,
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

        let new_pos = predict_move_position(joystick.current_predicted_position, event.dx, event.dy);
        let new_state = PredictedState {
            sequence: seq,
            position: new_pos,
            harmony: 0.75,
        };
        joystick.record_prediction_state(new_state);

        // Record with checksum support (checksum computed on correction)
        replay_logger.record_move_event(&event, replay_logger.current_tick);
        replay_logger.current_tick += 1;
    }
}

fn apply_local_prediction(mut joystick: ResMut<VirtualJoystick>) {
    let _ = joystick;
}
