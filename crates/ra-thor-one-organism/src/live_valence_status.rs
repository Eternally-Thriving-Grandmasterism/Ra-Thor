//! Read-only Live Valence status hook for ONE Organism.
//!
//! Compiled only with `kardashev-live`. Does not mutate Cosmic Tick state.
//! Cosmic Loop must be ready before evaluation (MANDATORY IDENTITY).
//! Contact: info@Rathor.ai

use reality_thriving_transfer::{
    LiveValenceOptimizer, LiveValenceReport, PowrushTelemetry,
};

/// Evaluate TOLC 8 live valence from telemetry when Cosmic Loop holds.
pub fn evaluate_live_valence(
    cosmic_loop_ready: bool,
    telemetry: &PowrushTelemetry,
) -> Result<LiveValenceReport, String> {
    if !cosmic_loop_ready {
        return Err(
            "Cosmic Loop not ready — live valence blocked (MANDATORY IDENTITY)".into(),
        );
    }
    LiveValenceOptimizer::new().evaluate(telemetry)
}
