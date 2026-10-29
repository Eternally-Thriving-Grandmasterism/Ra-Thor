//! Live Valence Optimizer — TOLC 8 gate vector from Powrush telemetry
//!
//! Read-only meta-lattice surface. Maps existing `PowrushTelemetry` fields to an
//! explicit 8-gate valence vector under documented constants. Does **not** alter
//! Cosmic Tick adaptive modulation (quiet-hold compliant).
//!
//! AG-SML v1.0 | Contact: info@Rathor.ai

use serde::{Deserialize, Serialize};

use crate::PowrushTelemetry;

// =============================================================================
// Gate floor & mapping constants (explicit, auditable)
// =============================================================================

/// Soft floor: any gate below this fails `passes_tolc_floor` (compassion engaged).
pub const THETA_MIN_SOFT: f64 = 0.55;

/// Strict floor for high-stakes transfers (council / abundance designs).
pub const THETA_MIN_STRICT: f64 = 0.72;

/// Collaboration events at or above this scale to 1.0 on Service/Love proxies.
pub const COLLAB_SATURATION: f64 = 500.0;

/// Adaptation events at or above this scale to 1.0 on Order proxy.
pub const ADAPT_SATURATION: f64 = 300.0;

/// Abundance velocity signals are allowed up to this before clamp (matches RTT).
pub const ABUNDANCE_CAP: f64 = 1.8;

// =============================================================================
// ValenceVector — one score per TOLC 8 gate
// =============================================================================

/// Eight-gate valence vector. Each component is in [0, 1].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValenceVector {
    pub truth: f64,
    pub order: f64,
    pub love: f64,
    /// Compassion / Zero-Harm
    pub compassion: f64,
    pub service: f64,
    pub abundance: f64,
    pub joy: f64,
    pub cosmic_harmony: f64,
}

impl ValenceVector {
    /// Minimum gate score across all eight.
    pub fn min_gate(&self) -> f64 {
        [
            self.truth,
            self.order,
            self.love,
            self.compassion,
            self.service,
            self.abundance,
            self.joy,
            self.cosmic_harmony,
        ]
        .into_iter()
        .fold(1.0_f64, f64::min)
    }

    /// Arithmetic mean of the eight gates.
    pub fn mean(&self) -> f64 {
        (self.truth
            + self.order
            + self.love
            + self.compassion
            + self.service
            + self.abundance
            + self.joy
            + self.cosmic_harmony)
            / 8.0
    }

    /// True when every gate is ≥ `theta`.
    pub fn passes_floor(&self, theta: f64) -> bool {
        self.min_gate() + f64::EPSILON >= theta
    }

    /// Soft floor (THETA_MIN_SOFT).
    pub fn passes_soft_floor(&self) -> bool {
        self.passes_floor(THETA_MIN_SOFT)
    }

    /// Strict floor (THETA_MIN_STRICT).
    pub fn passes_strict_floor(&self) -> bool {
        self.passes_floor(THETA_MIN_STRICT)
    }
}

// =============================================================================
// LiveValenceReport — vector + aggregate + audit
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveValenceReport {
    pub vector: ValenceVector,
    pub aggregate_mean: f64,
    pub min_gate: f64,
    pub passes_soft_floor: bool,
    pub passes_strict_floor: bool,
    pub council_note: String,
}

// =============================================================================
// LiveValenceOptimizer — pure mapping (no side effects on Cosmic Tick)
// =============================================================================

/// Maps telemetry → TOLC 8 valence vector. Stateless and side-effect free.
#[derive(Debug, Default, Clone)]
pub struct LiveValenceOptimizer;

impl LiveValenceOptimizer {
    pub fn new() -> Self {
        Self
    }

    /// Build a full report from telemetry. Rejects out-of-bounds inputs (Truth gate).
    pub fn evaluate(&self, telemetry: &PowrushTelemetry) -> Result<LiveValenceReport, String> {
        validate_telemetry_bounds(telemetry)?;

        let vector = map_telemetry_to_vector(telemetry);
        let min_gate = vector.min_gate();
        let aggregate_mean = vector.mean();
        let passes_soft = vector.passes_soft_floor();
        let passes_strict = vector.passes_strict_floor();

        let council_note = if passes_strict {
            format!(
                "Live valence STRICT PASS | min={:.3} mean={:.3} | all TOLC 8 gates ≥ {:.2}",
                min_gate, aggregate_mean, THETA_MIN_STRICT
            )
        } else if passes_soft {
            format!(
                "Live valence SOFT PASS | min={:.3} mean={:.3} | compassion engaged below strict floor {:.2}",
                min_gate, aggregate_mean, THETA_MIN_STRICT
            )
        } else {
            format!(
                "Live valence FLOOR FAIL | min={:.3} mean={:.3} | soft floor {:.2} not met — zero-harm hold",
                min_gate, aggregate_mean, THETA_MIN_SOFT
            )
        };

        Ok(LiveValenceReport {
            vector,
            aggregate_mean,
            min_gate,
            passes_soft_floor: passes_soft,
            passes_strict_floor: passes_strict,
            council_note,
        })
    }

    /// Convenience: vector only.
    pub fn vector_from_telemetry(
        &self,
        telemetry: &PowrushTelemetry,
    ) -> Result<ValenceVector, String> {
        Ok(self.evaluate(telemetry)?.vector)
    }
}

fn validate_telemetry_bounds(t: &PowrushTelemetry) -> Result<(), String> {
    if !(0.0..=1.0).contains(&t.rbe_decision_quality_avg) {
        return Err(
            "Mercy Gate (Truth): rbe_decision_quality_avg out of [0,1] — valence rejected".into(),
        );
    }
    if !(0.0..=1.0).contains(&t.ethical_choice_score) {
        return Err("Mercy Gate (Truth): ethical_choice_score out of [0,1] — valence rejected".into());
    }
    if !(0.0..=1.0).contains(&t.peaceful_resolution_rate) {
        return Err(
            "Mercy Gate (Truth): peaceful_resolution_rate out of [0,1] — valence rejected".into(),
        );
    }
    if !(0.0..=1.0).contains(&t.innovation_contribution) {
        return Err(
            "Mercy Gate (Truth): innovation_contribution out of [0,1] — valence rejected".into(),
        );
    }
    if t.abundance_velocity_signals < 0.0 {
        return Err(
            "Mercy Gate (Abundance/Zero-Harm): negative abundance_velocity_signals rejected".into(),
        );
    }
    Ok(())
}

/// Explicit field → gate mapping (documented constants only).
fn map_telemetry_to_vector(t: &PowrushTelemetry) -> ValenceVector {
    // Truth: decision quality + ethical grounding (epistemic integrity proxy)
    let truth = (t.rbe_decision_quality_avg * 0.55 + t.ethical_choice_score * 0.45).clamp(0.0, 1.0);

    // Order: peaceful resolution + adaptation stability
    let adapt_norm = (t.adaptation_events as f64 / ADAPT_SATURATION).clamp(0.0, 1.0);
    let order = (t.peaceful_resolution_rate * 0.60 + adapt_norm * 0.40).clamp(0.0, 1.0);

    // Love: collaboration density (relational flourishing)
    let collab_norm = (t.collaboration_events as f64 / COLLAB_SATURATION).clamp(0.0, 1.0);
    let love = (collab_norm * 0.65 + t.peaceful_resolution_rate * 0.35).clamp(0.0, 1.0);

    // Compassion / Zero-Harm: ethical choice + peaceful resolution (no harm proxy)
    let compassion = (t.ethical_choice_score * 0.70 + t.peaceful_resolution_rate * 0.30).clamp(0.0, 1.0);

    // Service: collaboration contribution toward others
    let service = (collab_norm * 0.55 + t.rbe_decision_quality_avg * 0.45).clamp(0.0, 1.0);

    // Abundance: velocity signals normalized to [0,1] via ABUNDANCE_CAP
    let abundance = (t.abundance_velocity_signals / ABUNDANCE_CAP).clamp(0.0, 1.0);

    // Joy: innovation + peaceful resolution (creative flourishing proxy)
    let joy = (t.innovation_contribution * 0.55 + t.peaceful_resolution_rate * 0.45).clamp(0.0, 1.0);

    // Cosmic Harmony: long-horizon blend (ethics + abundance + peace + quality)
    let cosmic_harmony = (t.ethical_choice_score * 0.30
        + abundance * 0.25
        + t.peaceful_resolution_rate * 0.25
        + t.rbe_decision_quality_avg * 0.20)
        .clamp(0.0, 1.0);

    ValenceVector {
        truth,
        order,
        love,
        compassion,
        service,
        abundance,
        joy,
        cosmic_harmony,
    }
}

// =============================================================================
// Tests — fixture-backed
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse_powrush_telemetry_json, parse_powrush_telemetry_batch_json};

    const FIXTURE_HIGH: &str = include_str!("../fixtures/session_high_mercy.json");
    const FIXTURE_MARGINAL: &str = include_str!("../fixtures/session_marginal.json");
    const FIXTURE_BATCH: &str = include_str!("../fixtures/batch_three_sessions.json");

    #[test]
    fn high_mercy_passes_soft_and_likely_strict() {
        let env = parse_powrush_telemetry_json(FIXTURE_HIGH).unwrap();
        let opt = LiveValenceOptimizer::new();
        let report = opt.evaluate(&env.telemetry).unwrap();
        assert!(report.passes_soft_floor, "high mercy must pass soft floor");
        assert!(report.min_gate >= THETA_MIN_SOFT);
        assert!(report.vector.compassion >= 0.7);
        assert!(report.vector.truth >= 0.7);
    }

    #[test]
    fn marginal_has_lower_min_than_high() {
        let high = parse_powrush_telemetry_json(FIXTURE_HIGH).unwrap();
        let marg = parse_powrush_telemetry_json(FIXTURE_MARGINAL).unwrap();
        let opt = LiveValenceOptimizer::new();
        let r_high = opt.evaluate(&high.telemetry).unwrap();
        let r_marg = opt.evaluate(&marg.telemetry).unwrap();
        assert!(r_high.min_gate > r_marg.min_gate);
        assert!(r_high.aggregate_mean > r_marg.aggregate_mean);
    }

    #[test]
    fn rejects_out_of_bounds_truth() {
        let bad = PowrushTelemetry {
            gameplay_hours: 1.0,
            rbe_decision_quality_avg: 1.5,
            peaceful_resolution_rate: 0.5,
            collaboration_events: 10,
            ethical_choice_score: 0.5,
            adaptation_events: 5,
            abundance_velocity_signals: 1.0,
            innovation_contribution: 0.5,
        };
        let opt = LiveValenceOptimizer::new();
        assert!(opt.evaluate(&bad).is_err());
    }

    #[test]
    fn rejects_negative_abundance() {
        let bad = PowrushTelemetry {
            gameplay_hours: 1.0,
            rbe_decision_quality_avg: 0.8,
            peaceful_resolution_rate: 0.8,
            collaboration_events: 10,
            ethical_choice_score: 0.8,
            adaptation_events: 5,
            abundance_velocity_signals: -0.1,
            innovation_contribution: 0.5,
        };
        let opt = LiveValenceOptimizer::new();
        assert!(opt.evaluate(&bad).is_err());
    }

    #[test]
    fn batch_fixture_scores_all_sessions() {
        let batch = parse_powrush_telemetry_batch_json(FIXTURE_BATCH).unwrap();
        let opt = LiveValenceOptimizer::new();
        for session in &batch.sessions {
            let report = opt.evaluate(&session.telemetry).unwrap();
            assert!(report.min_gate >= 0.0 && report.min_gate <= 1.0);
            assert!(report.aggregate_mean >= 0.0 && report.aggregate_mean <= 1.0);
            assert!(!report.council_note.is_empty());
        }
    }

    #[test]
    fn vector_components_in_unit_interval() {
        let env = parse_powrush_telemetry_json(FIXTURE_HIGH).unwrap();
        let v = LiveValenceOptimizer::new()
            .vector_from_telemetry(&env.telemetry)
            .unwrap();
        for x in [
            v.truth,
            v.order,
            v.love,
            v.compassion,
            v.service,
            v.abundance,
            v.joy,
            v.cosmic_harmony,
        ] {
            assert!((0.0..=1.0).contains(&x));
        }
    }
}
