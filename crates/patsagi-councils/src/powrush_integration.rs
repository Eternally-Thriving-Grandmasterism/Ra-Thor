//! # Powrush Integration Bridge — v14.15.5
//!
//! Single clean entry point the main Powrush simulation loop can call every cycle.
//! The 16 PATSAGi Councils automatically govern, propose, and implement world changes.
//!
//! Includes the closed dual-repo feedback loop + ResonanceMetrics observability.
//!
//! Living Cosmic Tick aligned.
//! Contact: info@Rathor.ai
//! AG-SML v1.0

use crate::feedback_loop::{PowrushTelemetrySnapshot, RaThorFeedbackLoop, FeedbackCycleResult};
use crate::observability::ResonanceMetrics;
use crate::{CouncilFocus, SimulationIntegration, WorldImpactType};
use powrush::PowrushGame;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const VERSION: &str = "14.15.5";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowrushPatsagiBridge {
    pub integration: SimulationIntegration,
    pub feedback: RaThorFeedbackLoop,
    pub enabled: bool,
    pub feedback_enabled: bool,
    pub version: &'static str,
    /// Last feedback cycle summary (for observability).
    #[serde(skip)]
    pub last_feedback: Option<FeedbackCycleResult>,
}

impl PowrushPatsagiBridge {
    pub fn new() -> Self {
        Self {
            integration: SimulationIntegration::new(),
            feedback: RaThorFeedbackLoop::new(),
            enabled: true,
            feedback_enabled: true,
            version: VERSION,
            last_feedback: None,
        }
    }

    /// Call once per simulation cycle from the main Powrush loop.
    /// Returns `Some(message)` if the Councils made a world change.
    pub async fn tick(&mut self, game: &mut PowrushGame) -> Option<String> {
        if !self.enabled {
            return None;
        }
        self.integration.tick(game).await
    }

    /// Full governance tick + optional soft feedback emission.
    ///
    /// After a successful Council intervention, builds a telemetry snapshot
    /// from the current game state and emits conservative policy hints
    /// if the mercy gates pass. Metrics are recorded automatically.
    pub async fn tick_with_feedback(
        &mut self,
        game: &mut PowrushGame,
        session_id: &str,
        export_seq: u64,
        emission_path: Option<&Path>,
    ) -> (Option<String>, Option<FeedbackCycleResult>) {
        let governance_msg = self.tick(game).await;

        let mut feedback_result = None;

        if self.feedback_enabled && governance_msg.is_some() {
            // Derive soft signals from current game state (conservative heuristics)
            let snap = self.build_telemetry_snapshot(game, session_id, export_seq);
            match self.feedback.deliberate_and_emit(&snap, emission_path) {
                Ok(res) => {
                    self.last_feedback = Some(res.clone());
                    feedback_result = Some(res);
                }
                Err(_) => {
                    // Silent on mercy-gate failure or no-hints — soft loop only
                    // Metrics already recorded inside deliberate_and_emit
                }
            }
        }

        (governance_msg, feedback_result)
    }

    /// Build a telemetry snapshot from live PowrushGame state.
    fn build_telemetry_snapshot(
        &self,
        game: &PowrushGame,
        session_id: &str,
        export_seq: u64,
    ) -> PowrushTelemetrySnapshot {
        let player_count = game.players.len().max(1) as f64;

        let avg_happiness = if !game.players.is_empty() {
            game.players.iter().map(|p| p.happiness as f64).sum::<f64>() / player_count
        } else {
            70.0
        } / 100.0;

        let avg_joy = if !game.players.is_empty() {
            game.players
                .iter()
                .map(|p| p.needs.joy as f64)
                .sum::<f64>()
                / player_count
        } else {
            65.0
        } / 100.0;

        // Simple proxies — keep soft and bounded
        let abundance = (avg_happiness * 0.6 + 0.3).clamp(0.2, 0.95);
        let peaceful = (avg_joy * 0.7 + 0.25).clamp(0.25, 0.95);
        let ethical = 0.78; // baseline high ethical floor for Ra-Thor
        let council = (self.integration.interventions as f64 * 0.02 + 0.35).clamp(0.2, 0.9);
        let innovation = (avg_happiness * 0.45 + 0.3).clamp(0.2, 0.9);
        let mercy = (avg_joy * 0.55 + 0.4).clamp(0.35, 0.97);

        PowrushTelemetrySnapshot {
            session_id: session_id.to_string(),
            export_seq,
            abundance_signal: abundance,
            peaceful_resolution_signal: peaceful,
            ethical_floor_signal: ethical,
            council_participation_signal: council,
            innovation_signal: innovation,
            mercy_presence_signal: mercy,
        }
    }

    /// Enable or disable Council governance.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Enable or disable the soft feedback emission loop.
    pub fn set_feedback_enabled(&mut self, enabled: bool) {
        self.feedback_enabled = enabled;
        self.feedback.enabled = enabled;
    }

    /// Current status for debugging or UI.
    pub fn get_status(&self) -> String {
        format!(
            "PowrushPatsagiBridge v{} | enabled={} | feedback={} | {}",
            self.version,
            self.enabled,
            self.feedback_enabled,
            self.integration.get_status()
        )
    }

    /// Force an immediate governance cycle (special events).
    pub async fn force_governance_cycle(&mut self, game: &mut PowrushGame) -> String {
        self.integration
            .governance_engine
            .propose_and_approve_world_change(
                CouncilFocus::EternalCompassion,
                "Forced Governance Cycle",
                "A special event has triggered immediate Council governance.",
                WorldImpactType::AllianceFormed,
                game,
            )
            .await
            .unwrap_or_else(|e| format!("Force governance error: {}", e))
    }

    /// Compact telemetry + metrics summary.
    pub fn summary(&self) -> String {
        let fb = self
            .last_feedback
            .as_ref()
            .map(|r| format!(" | last_hints={}", r.hints_emitted))
            .unwrap_or_default();
        format!(
            "PowrushPatsagiBridge v{} | enabled={} | feedback={} | interventions={}{} | {}",
            self.version,
            self.enabled,
            self.feedback_enabled,
            self.integration.interventions,
            fb,
            self.feedback.metrics_summary()
        )
    }

    /// Direct access to the observability surface.
    pub fn metrics(&self) -> &ResonanceMetrics {
        &self.feedback.metrics
    }

    pub fn metrics_summary(&self) -> String {
        self.feedback.metrics_summary()
    }
}

impl Default for PowrushPatsagiBridge {
    fn default() -> Self {
        Self::new()
    }
}
