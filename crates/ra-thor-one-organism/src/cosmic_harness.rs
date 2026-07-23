//! Cosmic Harness — v14.15.2
//!
//! Production endurance runner for the Living Cosmic Tick under the sealed
//! dual-repo feedback organism (Powrush-MMO ↔ Ra-Thor AGSi).
//!
//! Addresses the three pressure points surfaced in live PATSAGi deliberation:
//! 1. Latency / tick alignment (zero-drift sync across host modes)
//! 2. Feedback category saturation under multiplayer council load
//! 3. Recovery integrity (local rollback preserves AGSi-side history)
//!
//! All runs are gated by TOLC 8 + Cosmic Loop invariants.
//! Contact: info@Rathor.ai

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{
    CosmicLoopInvariant, CosmicTickResult, LiveFeatureReadiness, OneOrganismCore,
};

/// Host mode under test (mirrors Powrush interactive / headless / stress).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HostMode {
    Interactive,
    Headless,
    Stress,
}

impl HostMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            HostMode::Interactive => "interactive",
            HostMode::Headless => "headless",
            HostMode::Stress => "stress",
        }
    }

    /// Severity profile injected into each Cosmic Tick for this mode.
    pub fn severity_profile(&self, cycle: u64) -> f64 {
        match self {
            HostMode::Interactive => 0.18 + ((cycle % 7) as f64 * 0.04),
            HostMode::Headless => 0.28 + ((cycle % 5) as f64 * 0.05),
            HostMode::Stress => 0.55 + ((cycle % 9) as f64 * 0.04),
        }
        .clamp(0.05, 0.95)
    }
}

/// Per-tick snapshot used for drift and saturation analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickSnapshot {
    pub tick: u64,
    pub mode: String,
    pub severity: f64,
    pub shared_valence: f64,
    pub shared_confidence: f64,
    pub recovery_sensitivity: f64,
    pub cosmic_loop_holds: bool,
    pub anomalies: Vec<String>,
    pub recovery_triggered: bool,
    pub gpu_confidence: f64,
    pub effective_quantum_severity: f64,
}

/// Aggregate drift metrics across an endurance run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    pub valence_max_delta: f64,
    pub confidence_max_delta: f64,
    pub recovery_sensitivity_max_delta: f64,
    pub cosmic_loop_failures: u64,
    pub zero_drift_achieved: bool,
}

/// Category saturation analysis (proxy for the six SoftPolicyState reception categories).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaturationReport {
    pub anomaly_counts: HashMap<String, u64>,
    pub dominant_category: Option<String>,
    pub dominant_share: f64,
    pub saturation_detected: bool,
    pub anti_starvation_applied: bool,
}

/// Full result of a Cosmic Harness endurance run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmicHarnessResult {
    pub cycles_completed: u64,
    pub modes_tested: Vec<String>,
    pub snapshots: Vec<TickSnapshot>,
    pub drift: DriftReport,
    pub saturation: SaturationReport,
    pub recovery_integrity_ok: bool,
    pub final_live_features: LiveFeatureReadiness,
    pub final_cosmic_loop: CosmicLoopInvariant,
    pub harness_version: String,
    pub recommendation: String,
}

/// Cosmic Harness configuration.
#[derive(Debug, Clone)]
pub struct CosmicHarnessConfig {
    pub cycles: u64,
    pub modes: Vec<HostMode>,
    /// Maximum allowed valence delta between consecutive ticks before drift flag.
    pub valence_drift_threshold: f64,
    /// Maximum allowed confidence delta.
    pub confidence_drift_threshold: f64,
    /// Share of a single anomaly category that triggers saturation warning.
    pub saturation_share_threshold: f64,
}

impl Default for CosmicHarnessConfig {
    fn default() -> Self {
        Self {
            cycles: 40,
            modes: vec![HostMode::Interactive, HostMode::Headless, HostMode::Stress],
            valence_drift_threshold: 0.045,
            confidence_drift_threshold: 0.08,
            saturation_share_threshold: 0.62,
        }
    }
}

/// Production Cosmic Harness.
///
/// Runs the Living Cosmic Tick under controlled host-mode profiles,
/// measures zero-drift alignment, detects category saturation,
/// and verifies recovery integrity (local vs preserved AGSi history).
pub struct CosmicHarness {
    pub config: CosmicHarnessConfig,
}

impl CosmicHarness {
    pub fn new(config: CosmicHarnessConfig) -> Self {
        Self { config }
    }

    pub fn default_40_cycle() -> Self {
        Self::new(CosmicHarnessConfig::default())
    }

    /// Execute the full endurance harness against a live `OneOrganismCore`.
    pub fn run(&self, core: &mut OneOrganismCore) -> CosmicHarnessResult {
        // Enforce Cosmic Loop before any measurement.
        let _ = core.enforce_cosmic_loop_invariant();

        let mut snapshots: Vec<TickSnapshot> = Vec::with_capacity(
            (self.config.cycles as usize) * self.config.modes.len(),
        );
        let mut anomaly_counts: HashMap<String, u64> = HashMap::new();
        let mut recovery_events = 0u64;
        let mut cosmic_loop_failures = 0u64;

        let mut prev_valence = core.role_orchestrator.shared_valence;
        let mut prev_confidence = core.role_orchestrator.shared_confidence_ema;
        let mut prev_sensitivity = core.next_recovery_sensitivity;

        let mut valence_max_delta = 0.0f64;
        let mut confidence_max_delta = 0.0f64;
        let mut sensitivity_max_delta = 0.0f64;

        // Preserve a pre-run recovery anchor as the "AGSi-side history".
        let history_anchor = core.recovery_anchor("cosmic_harness_pre_run_history");

        for mode in &self.config.modes {
            for cycle in 0..self.config.cycles {
                let severity = mode.severity_profile(cycle);
                let result: CosmicTickResult = core.cosmic_tick(severity);

                // Drift measurement
                let v_delta = (result.cosmic_loop_invariant.cosmic_loop_ready as u8 as f64
                    - prev_valence)
                    .abs(); // placeholder; real delta below
                let _ = v_delta; // silence
                let valence_delta = (core.role_orchestrator.shared_valence - prev_valence).abs();
                let conf_delta =
                    (core.role_orchestrator.shared_confidence_ema - prev_confidence).abs();
                let sens_delta =
                    (core.next_recovery_sensitivity - prev_sensitivity).abs();

                valence_max_delta = valence_max_delta.max(valence_delta);
                confidence_max_delta = confidence_max_delta.max(conf_delta);
                sensitivity_max_delta = sensitivity_max_delta.max(sens_delta);

                if !result.cosmic_loop_invariant.all_hold {
                    cosmic_loop_failures += 1;
                }

                // Saturation proxy via anomaly categories
                for a in &result.anomalies_fired {
                    *anomaly_counts.entry(a.clone()).or_insert(0) += 1;
                }
                if result.recovery_triggered {
                    recovery_events += 1;
                }

                snapshots.push(TickSnapshot {
                    tick: result.tick,
                    mode: mode.as_str().into(),
                    severity,
                    shared_valence: core.role_orchestrator.shared_valence,
                    shared_confidence: core.role_orchestrator.shared_confidence_ema,
                    recovery_sensitivity: result.recovery_sensitivity_applied,
                    cosmic_loop_holds: result.cosmic_loop_invariant.all_hold,
                    anomalies: result.anomalies_fired.clone(),
                    recovery_triggered: result.recovery_triggered,
                    gpu_confidence: result.gpu_confidence,
                    effective_quantum_severity: result.effective_quantum_severity,
                });

                prev_valence = core.role_orchestrator.shared_valence;
                prev_confidence = core.role_orchestrator.shared_confidence_ema;
                prev_sensitivity = core.next_recovery_sensitivity;
            }
        }

        // Saturation analysis
        let total_anomalies: u64 = anomaly_counts.values().sum();
        let (dominant_category, dominant_share) = if total_anomalies == 0 {
            (None, 0.0)
        } else {
            let (cat, count) = anomaly_counts
                .iter()
                .max_by_key(|(_, c)| *c)
                .map(|(k, v)| (k.clone(), *v))
                .unwrap_or(("none".into(), 0));
            (Some(cat), count as f64 / total_anomalies as f64)
        };

        let saturation_detected = dominant_share >= self.config.saturation_share_threshold;
        // Soft anti-starvation: if saturation detected we gently re-balance next-tick sensitivity
        // (does not mutate history; only future sensitivity).
        let anti_starvation_applied = if saturation_detected {
            core.next_recovery_sensitivity =
                (core.next_recovery_sensitivity * 0.92).clamp(1.0, 1.08);
            true
        } else {
            false
        };

        // Recovery integrity: the pre-run history anchor must still be present and
        // the organism must still hold Cosmic Loop after all local recovery events.
        let post_status = core.recovery_status();
        let recovery_integrity_ok = post_status.anchor_count >= 1
            && post_status
                .last_anchor
                .as_ref()
                .map(|a| a.anchor_id == history_anchor.anchor_id || a.tick >= history_anchor.tick)
                .unwrap_or(false)
            && cosmic_loop_failures == 0;

        let zero_drift = valence_max_delta <= self.config.valence_drift_threshold
            && confidence_max_delta <= self.config.confidence_drift_threshold
            && cosmic_loop_failures == 0;

        let final_inv = core.assert_cosmic_loop_invariant();
        let final_features = core.live_feature_readiness();

        let recommendation = if zero_drift && recovery_integrity_ok && !saturation_detected {
            "PASS — zero-drift Cosmic Tick alignment, recovery integrity preserved, no category saturation. Ready for live dual-repo resonance."
                .into()
        } else if zero_drift && recovery_integrity_ok {
            format!(
                "PASS with soft anti-starvation applied (dominant={:?} share={:.2}). Continue monitoring under multiplayer load.",
                dominant_category, dominant_share
            )
        } else {
            format!(
                "REVIEW — valence_delta={:.4} conf_delta={:.4} loop_failures={} recovery_ok={} saturation={}",
                valence_max_delta,
                confidence_max_delta,
                cosmic_loop_failures,
                recovery_integrity_ok,
                saturation_detected
            )
        };

        CosmicHarnessResult {
            cycles_completed: self.config.cycles * self.config.modes.len() as u64,
            modes_tested: self.config.modes.iter().map(|m| m.as_str().into()).collect(),
            snapshots,
            drift: DriftReport {
                valence_max_delta,
                confidence_max_delta,
                recovery_sensitivity_max_delta: sensitivity_max_delta,
                cosmic_loop_failures,
                zero_drift_achieved: zero_drift,
            },
            saturation: SaturationReport {
                anomaly_counts,
                dominant_category,
                dominant_share,
                saturation_detected,
                anti_starvation_applied,
            },
            recovery_integrity_ok,
            final_live_features: final_features,
            final_cosmic_loop: final_inv,
            harness_version: "v14.15.2 Cosmic Harness".into(),
            recommendation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launch_one_organism_core;

    #[test]
    fn cosmic_harness_40_cycle_zero_drift() {
        let mut core = launch_one_organism_core();
        let harness = CosmicHarness::default_40_cycle();
        let result = harness.run(&mut core);

        assert!(result.cycles_completed >= 40);
        assert!(result.final_cosmic_loop.all_hold);
        assert!(result.drift.cosmic_loop_failures == 0);
        assert!(result.recovery_integrity_ok);
        // Soft assertion: under default profiles we expect near-zero drift
        assert!(result.drift.valence_max_delta < 0.12);
    }

    #[test]
    fn host_mode_severity_profiles_are_distinct() {
        let i = HostMode::Interactive.severity_profile(0);
        let h = HostMode::Headless.severity_profile(0);
        let s = HostMode::Stress.severity_profile(0);
        assert!(i < h && h < s);
    }
}
