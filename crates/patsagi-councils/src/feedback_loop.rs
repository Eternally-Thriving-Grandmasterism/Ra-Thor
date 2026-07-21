//! crates/patsagi-councils/src/feedback_loop.rs — v14.15.5
//! Ra-Thor Feedback Loop — closed dual-repo soft RTT with Powrush-MMO
//!
//! Flow:
//!   Powrush telemetry snapshot
//!     → lightweight PATSAGi-style deliberation (mercy-gated)
//!     → 0..N conservative PolicyHints (closed 6-category set)
//!     → atomic emission via powrush::policy_hint_emission
//!
//! Only emits after a successful mercy-passing deliberation.
//! Prefer small deltas. Offline-first. Zero-harm.
//!
//! Now reports into ResonanceMetrics for observability.
//!
//! Canonical emission contract: Powrush-MMO/docs/RA_THOR_POLICY_HINT_EMISSION.md
//! Contact: info@Rathor.ai
//! TOLC 8 + PATSAGi | Living Cosmic Tick

use crate::observability::{BlockReason, ResonanceMetrics};
use powrush::{
    PolicyHint, emit_after_deliberation, EmissionError, DEFAULT_EMISSION_PATH,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

/// High-level signals extracted from Powrush telemetry / session state.
/// All values expected in [0.0, 1.0] where meaningful.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowrushTelemetrySnapshot {
    pub session_id: String,
    pub export_seq: u64,
    /// Overall abundance / resource health signal
    pub abundance_signal: f64,
    /// Peace / conflict tension (higher = more peaceful)
    pub peaceful_resolution_signal: f64,
    /// Ethical / zero-harm floor adherence
    pub ethical_floor_signal: f64,
    /// Active council / governance participation density
    pub council_participation_signal: f64,
    /// Innovation / creative activity level
    pub innovation_signal: f64,
    /// Ambient mercy / compassion presence
    pub mercy_presence_signal: f64,
}

impl Default for PowrushTelemetrySnapshot {
    fn default() -> Self {
        Self {
            session_id: "*".to_string(),
            export_seq: 0,
            abundance_signal: 0.5,
            peaceful_resolution_signal: 0.5,
            ethical_floor_signal: 0.7,
            council_participation_signal: 0.4,
            innovation_signal: 0.4,
            mercy_presence_signal: 0.6,
        }
    }
}

#[derive(Debug, Error)]
pub enum FeedbackError {
    #[error("emission error: {0}")]
    Emission(#[from] EmissionError),
    #[error("deliberation did not pass mercy gates — no emission")]
    MercyGateFailed,
    #[error("no actionable soft hints generated")]
    NoHints,
}

/// Result of a single feedback cycle.
#[derive(Debug, Clone)]
pub struct FeedbackCycleResult {
    pub session_id: String,
    pub export_seq: u64,
    pub hints_emitted: usize,
    pub emission_path: String,
    pub mercy_passed: bool,
    pub summary: String,
}

/// Core feedback loop engine.
#[derive(Debug, Clone)]
pub struct RaThorFeedbackLoop {
    /// Minimum average signal strength required to consider emission.
    pub min_signal_threshold: f64,
    /// Maximum number of hints per cycle (conservative).
    pub max_hints_per_cycle: usize,
    /// Base strength scaling (keep soft).
    pub strength_scale: f64,
    /// Whether emission is enabled.
    pub enabled: bool,
    /// Lightweight observability.
    pub metrics: ResonanceMetrics,
}

impl Default for RaThorFeedbackLoop {
    fn default() -> Self {
        Self {
            min_signal_threshold: 0.55,
            max_hints_per_cycle: 4,
            strength_scale: 0.45,
            enabled: true,
            metrics: ResonanceMetrics::new(),
        }
    }
}

impl RaThorFeedbackLoop {
    pub fn new() -> Self {
        Self::default()
    }

    /// Full cycle: deliberate on telemetry → produce conservative hints → emit.
    ///
    /// Only emits when the internal mercy check passes and at least one
    /// actionable soft hint is generated.
    pub fn deliberate_and_emit(
        &mut self,
        telemetry: &PowrushTelemetrySnapshot,
        emission_path: Option<&Path>,
    ) -> Result<FeedbackCycleResult, FeedbackError> {
        if !self.enabled {
            self.metrics.record_emission_block(BlockReason::MercyGate);
            return Err(FeedbackError::MercyGateFailed);
        }

        // Simple aggregate mercy gate (conservative)
        let avg_signal = (
            telemetry.abundance_signal
                + telemetry.peaceful_resolution_signal
                + telemetry.ethical_floor_signal
                + telemetry.council_participation_signal
                + telemetry.innovation_signal
                + telemetry.mercy_presence_signal
        ) / 6.0;

        let mercy_passed = avg_signal >= self.min_signal_threshold
            && telemetry.ethical_floor_signal >= 0.55
            && telemetry.mercy_presence_signal >= 0.45;

        if !mercy_passed {
            self.metrics.record_emission_block(BlockReason::MercyGate);
            return Err(FeedbackError::MercyGateFailed);
        }

        let mut hints = Vec::new();

        // Map signals → closed-category soft hints (prefer small deltas)
        self.maybe_push_hint(
            &mut hints,
            "abundance_bias",
            telemetry.abundance_signal,
            0.04,
            format!(
                "PATSAGi abundance signal {:.2} — gentle resource flow encouragement",
                telemetry.abundance_signal
            ),
        );
        self.maybe_push_hint(
            &mut hints,
            "peaceful_resolution_weight",
            telemetry.peaceful_resolution_signal,
            0.05,
            format!(
                "Peace signal {:.2} — soft weight toward non-violent resolution paths",
                telemetry.peaceful_resolution_signal
            ),
        );
        self.maybe_push_hint(
            &mut hints,
            "ethical_floor",
            telemetry.ethical_floor_signal,
            0.03,
            format!(
                "Ethical floor {:.2} — reinforce zero-harm baseline",
                telemetry.ethical_floor_signal
            ),
        );
        self.maybe_push_hint(
            &mut hints,
            "council_participation_nudge",
            telemetry.council_participation_signal,
            0.04,
            format!(
                "Council density {:.2} — gentle participation invitation",
                telemetry.council_participation_signal
            ),
        );
        self.maybe_push_hint(
            &mut hints,
            "innovation_encouragement",
            telemetry.innovation_signal,
            0.035,
            format!(
                "Innovation signal {:.2} — soft creative encouragement",
                telemetry.innovation_signal
            ),
        );
        self.maybe_push_hint(
            &mut hints,
            "mercy_presence",
            telemetry.mercy_presence_signal,
            0.06,
            format!(
                "Mercy presence {:.2} — ambient compassion uplift (TOLC 8)",
                telemetry.mercy_presence_signal
            ),
        );

        // Cap to keep soft
        if hints.len() > self.max_hints_per_cycle {
            hints.truncate(self.max_hints_per_cycle);
        }

        if hints.is_empty() {
            self.metrics.record_emission_block(BlockReason::NoHints);
            return Err(FeedbackError::NoHints);
        }

        let path = emit_after_deliberation(
            &telemetry.session_id,
            telemetry.export_seq,
            hints.clone(),
            emission_path,
        )?;

        self.metrics.record_emission_success(hints.len());

        let summary = format!(
            "Ra-Thor Feedback Cycle | session={} seq={} | hints={} | avg_signal={:.3} | path={}",
            telemetry.session_id,
            telemetry.export_seq,
            hints.len(),
            avg_signal,
            path.display()
        );

        Ok(FeedbackCycleResult {
            session_id: telemetry.session_id.clone(),
            export_seq: telemetry.export_seq,
            hints_emitted: hints.len(),
            emission_path: path.display().to_string(),
            mercy_passed: true,
            summary,
        })
    }

    fn maybe_push_hint(
        &self,
        hints: &mut Vec<PolicyHint>,
        category: &str,
        signal: f64,
        base_delta: f64,
        rationale: String,
    ) {
        if signal < self.min_signal_threshold {
            return;
        }

        let strength = (signal * self.strength_scale).clamp(0.15, 0.75);
        let delta = (base_delta * signal).clamp(0.01, 0.12);
        let mercy_factor = (0.85 + signal * 0.12).clamp(0.85, 0.99);

        let hint_id = format!(
            "ra-thor-{}-{}-{}",
            category,
            (signal * 1000.0) as u64,
            hints.len()
        );

        if let Ok(h) = PolicyHint::new(
            hint_id,
            category,
            strength,
            mercy_factor,
            delta,
            Some(rationale),
            None,
        ) {
            hints.push(h);
        }
    }

    /// Convenience: emit a single conservative mercy-presence hint.
    pub fn emit_mercy_nudge(
        &mut self,
        session_id: &str,
        export_seq: u64,
        strength: f64,
        rationale: &str,
        path: Option<&Path>,
    ) -> Result<FeedbackCycleResult, FeedbackError> {
        let hint = powrush::conservative_mercy_presence(
            format!("mercy-nudge-{}", export_seq),
            strength,
            0.04,
            rationale,
        )?;

        let written = emit_after_deliberation(session_id, export_seq, vec![hint], path)?;
        self.metrics.record_emission_success(1);

        Ok(FeedbackCycleResult {
            session_id: session_id.to_string(),
            export_seq,
            hints_emitted: 1,
            emission_path: written.display().to_string(),
            mercy_passed: true,
            summary: format!(
                "Mercy nudge emitted | session={} seq={} | path={}",
                session_id, export_seq, written.display()
            ),
        })
    }

    /// Current observability snapshot.
    pub fn metrics_summary(&self) -> String {
        self.metrics.summary()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    #[test]
    fn feedback_emits_on_strong_signals() {
        let mut loop_ = RaThorFeedbackLoop::new();
        let mut snap = PowrushTelemetrySnapshot::default();
        snap.session_id = "test-sess-001".into();
        snap.export_seq = 99;
        snap.abundance_signal = 0.82;
        snap.peaceful_resolution_signal = 0.78;
        snap.ethical_floor_signal = 0.91;
        snap.mercy_presence_signal = 0.88;
        snap.council_participation_signal = 0.65;
        snap.innovation_signal = 0.70;

        let mut path = env::temp_dir();
        path.push(format!("ra_thor_fb_test_{}.json", std::process::id()));

        let result = loop_.deliberate_and_emit(&snap, Some(&path)).unwrap();
        assert!(result.mercy_passed);
        assert!(result.hints_emitted > 0);
        assert!(path.exists());
        assert_eq!(loop_.metrics.emission_successes, 1);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn low_signals_blocked_by_mercy_gate() {
        let mut loop_ = RaThorFeedbackLoop::new();
        let mut snap = PowrushTelemetrySnapshot::default();
        snap.ethical_floor_signal = 0.3;
        snap.mercy_presence_signal = 0.2;

        let err = loop_.deliberate_and_emit(&snap, None);
        assert!(matches!(err, Err(FeedbackError::MercyGateFailed)));
        assert_eq!(loop_.metrics.mercy_gate_failures, 1);
    }
}
