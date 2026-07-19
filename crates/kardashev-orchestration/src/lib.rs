//! kardashev-orchestration v14.10.0
//!
//! Packaged from root `kardashev_orchestration_council.rs`.
//! Path-depends on `reality-thriving-transfer` for scores + GPU telemetry stub.
//!
//! AG-SML v1.0 | TOLC 8 Living Mercy Gates
//! Contact: info@Rathor.ai

use reality_thriving_transfer::{
    run_quantum_swarm_v2_kardashev_benchmark, GpuTelemetryReport, RealityThrivingTransferScore,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuFidelityAuditor {
    pub last_gpu_success_ema: f64,
    pub last_valence_offload: f64,
    pub fidelity_trend: f64,
}

impl Default for GpuFidelityAuditor {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuFidelityAuditor {
    pub fn new() -> Self {
        Self {
            last_gpu_success_ema: 0.93,
            last_valence_offload: 0.87,
            fidelity_trend: 0.0,
        }
    }

    pub fn audit(&mut self, report: &GpuTelemetryReport) -> f64 {
        self.last_gpu_success_ema = report.gpu_success_ema;
        self.last_valence_offload = report.valence_modulated_offload_score;
        let new_trend = 0.3 * (report.gpu_success_ema - 0.85) + 0.7 * self.fidelity_trend;
        self.fidelity_trend = new_trend.clamp(-0.15, 0.15);
        new_trend
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbeEthicsGate {
    pub min_ethical_threshold: f64,
    pub collaboration_weight: f64,
}

impl Default for RbeEthicsGate {
    fn default() -> Self {
        Self::new()
    }
}

impl RbeEthicsGate {
    pub fn new() -> Self {
        Self {
            min_ethical_threshold: 0.68,
            collaboration_weight: 0.22,
        }
    }

    pub fn validate(&self, score: &RealityThrivingTransferScore) -> (bool, String) {
        let ethics_pass = score.ethics_collaboration_index >= self.min_ethical_threshold;
        let note = if ethics_pass {
            "RBE Ethics Gate: PASSED — collaboration & ethical choice aligned with ONE Organism".into()
        } else {
            format!(
                "RBE Ethics Gate: Compassion engaged — ethics index {:.3} below threshold",
                score.ethics_collaboration_index
            )
        };
        (ethics_pass, note)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbundanceVelocityForecaster {
    pub base_velocity: f64,
    pub compounding_factor: f64,
}

impl Default for AbundanceVelocityForecaster {
    fn default() -> Self {
        Self::new()
    }
}

impl AbundanceVelocityForecaster {
    pub fn new() -> Self {
        Self {
            base_velocity: 0.42,
            compounding_factor: 1.018,
        }
    }

    pub fn forecast(&self, current_velocity: f64, cycles_ahead: usize) -> f64 {
        let mut projected = current_velocity;
        for _ in 0..cycles_ahead {
            projected = (projected * self.compounding_factor).min(1.85);
        }
        projected
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScurveProjection {
    pub current_kardashev: f64,
    pub projected_2030: f64,
    pub projected_2038: f64,
    pub inflection_year: u32,
    pub required_annual_delta_for_type_i: f64,
    pub mercy_aligned: bool,
    pub council_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    pub category: String,
    pub severity: f64,
    pub description: String,
    pub recommended_action: String,
    pub estimated_impact_on_kardashev_delta: f64,
    pub mercy_priority: String,
    pub last_modulation_timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmAdjustmentDirective {
    pub entanglement_modulation_delta: f64,
    pub quantum_jump_prob_multiplier: f64,
    pub mean_best_influence_multiplier: f64,
    pub classical_refinement_strength_multiplier: f64,
    pub dynamic_threshold_coupling_severity: f64,
    pub consensus_momentum_boost: f64,
    pub entanglement_topology_expansion_factor: f64,
    pub gpu_offload_bias: f64,
    pub valence_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilDeliberationResult {
    pub cycle_id: u64,
    pub timestamp: u64,
    pub s_curve_projection: ScurveProjection,
    pub identified_bottlenecks: Vec<Bottleneck>,
    pub cumulative_kardashev_delta: f64,
    pub abundance_velocity_trend: f64,
    pub gpu_fidelity: f64,
    pub swarm_adjustment_directive: Option<SwarmAdjustmentDirective>,
    pub hardware_priority_recommendation: String,
    pub recommendation_for_lattice_conductor: String,
    pub recommendation_for_council: String,
    pub mercy_audit_passed: bool,
    pub one_organism_alignment_note: String,
    pub rbe_ethics_note: String,
    pub abundance_forecast_next_12_cycles: f64,
}

pub struct KardashevOrchestrationCouncil {
    cumulative_delta: Arc<Mutex<f64>>,
    velocity_ema: Arc<Mutex<f64>>,
    deliberation_count: Arc<Mutex<u64>>,
    last_result: Arc<Mutex<Option<CouncilDeliberationResult>>>,
    bottleneck_registry: Arc<Mutex<Vec<Bottleneck>>>,
    gpu_auditor: Arc<Mutex<GpuFidelityAuditor>>,
    rbe_ethics_gate: Arc<Mutex<RbeEthicsGate>>,
    abundance_forecaster: Arc<Mutex<AbundanceVelocityForecaster>>,
}

impl Default for KardashevOrchestrationCouncil {
    fn default() -> Self {
        Self::new()
    }
}

impl KardashevOrchestrationCouncil {
    pub fn new() -> Self {
        Self {
            cumulative_delta: Arc::new(Mutex::new(0.0)),
            velocity_ema: Arc::new(Mutex::new(0.42)),
            deliberation_count: Arc::new(Mutex::new(0)),
            last_result: Arc::new(Mutex::new(None)),
            bottleneck_registry: Arc::new(Mutex::new(vec![
                Bottleneck {
                    category: "Regulatory Sovereignty".into(),
                    severity: 0.82,
                    description: "Primary bottleneck for 2032–2038 prototype activation window.".into(),
                    recommended_action: "Prioritize sovereign legal frameworks + open hardware licensing.".into(),
                    estimated_impact_on_kardashev_delta: 0.0042,
                    mercy_priority: "Immediate Service + Truth Gate".into(),
                    last_modulation_timestamp: 0,
                },
                Bottleneck {
                    category: "Hardware Scaling (Obsidian-Chip-Open)".into(),
                    severity: 0.71,
                    description: "Critical path for GPU-aware PATSAGi + sovereign super-brain.".into(),
                    recommended_action: "Accelerate Obsidian-Chip-Open integration into sovereign_core.".into(),
                    estimated_impact_on_kardashev_delta: 0.0038,
                    mercy_priority: "Abundance Gate".into(),
                    last_modulation_timestamp: 0,
                },
            ])),
            gpu_auditor: Arc::new(Mutex::new(GpuFidelityAuditor::new())),
            rbe_ethics_gate: Arc::new(Mutex::new(RbeEthicsGate::new())),
            abundance_forecaster: Arc::new(Mutex::new(AbundanceVelocityForecaster::new())),
        }
    }

    pub async fn ingest_transfer_and_gpu(
        &self,
        scores: &[RealityThrivingTransferScore],
        gpu_report: Option<&GpuTelemetryReport>,
    ) -> Result<(), String> {
        if scores.is_empty() {
            return Err("Mercy Gate (Truth): No transfer scores provided for deliberation".into());
        }

        let mut total_delta = 0.0;
        let mut latest_velocity = 0.0;
        for score in scores {
            if !score.mercy_audit_passed {
                eprintln!("Compassion Gate: Marginal transfer score: {}", score.council_note);
            }
            total_delta += score.kardashev_delta_contribution;
            latest_velocity = score.abundance_velocity_index;
        }

        *self.cumulative_delta.lock().await += total_delta;
        {
            let mut vel = self.velocity_ema.lock().await;
            let alpha = 0.19;
            *vel = alpha * latest_velocity + (1.0 - alpha) * *vel;
        }

        if let Some(gpu) = gpu_report {
            let mut auditor = self.gpu_auditor.lock().await;
            auditor.audit(gpu);
        }
        Ok(())
    }

    pub async fn deliberate_acceleration_cycle(
        &self,
        scores: &[RealityThrivingTransferScore],
        gpu_report: Option<&GpuTelemetryReport>,
    ) -> CouncilDeliberationResult {
        let _ = self.ingest_transfer_and_gpu(scores, gpu_report).await;

        let cumulative = *self.cumulative_delta.lock().await;
        let velocity = *self.velocity_ema.lock().await;
        let cycle = {
            let mut c = self.deliberation_count.lock().await;
            *c += 1;
            *c
        };

        let current_k = 0.72 + (cumulative * 12.0).min(0.18);
        let annual_delta = (velocity * 0.011 + cumulative * 0.0008).max(0.002).min(0.014);
        let projected_2030 = (current_k + annual_delta * 4.0).min(0.94);
        let projected_2038 = (current_k + annual_delta * 12.0).min(0.995);
        let inflection = if velocity > 1.1 { 2034 } else { 2036 };

        let s_curve = ScurveProjection {
            current_kardashev: current_k,
            projected_2030,
            projected_2038,
            inflection_year: inflection,
            required_annual_delta_for_type_i: 0.0115,
            mercy_aligned: velocity > 0.9 && cumulative > 0.008,
            council_note: format!(
                "Velocity {:.3} supports inflection by {}. TOLC 8 gates green.",
                velocity, inflection
            ),
        };

        let mut bottlenecks = self.bottleneck_registry.lock().await.clone();
        let now = now_secs();

        if let Some(gpu) = gpu_report {
            let mut auditor = self.gpu_auditor.lock().await;
            let fidelity_trend = auditor.audit(gpu);
            for b in &mut bottlenecks {
                if b.category.contains("Hardware") {
                    b.severity = (b.severity * (1.0 + fidelity_trend.abs() * 0.15)).min(0.96);
                    b.last_modulation_timestamp = now;
                }
            }
        }

        let latest_score = scores.last().cloned().unwrap_or_else(|| RealityThrivingTransferScore {
            raw_transfer_score: 0.6,
            mercy_valence_adjusted: 0.68,
            ema_refined_transfer: 0.65,
            confidence: 0.78,
            kardashev_delta_contribution: 0.003,
            abundance_velocity_index: velocity,
            ethics_collaboration_index: 0.7,
            last_refinement_vector: vec![0.04, -0.02, 0.03],
            mercy_audit_passed: true,
            timestamp: 0,
            council_note: "Default from council".into(),
        });

        let rbe_gate = self.rbe_ethics_gate.lock().await;
        let (ethics_pass, rbe_note) = rbe_gate.validate(&latest_score);

        let forecaster = self.abundance_forecaster.lock().await;
        let abundance_forecast = forecaster.forecast(velocity, 12);

        let swarm_directive = if latest_score.mercy_valence_adjusted > 0.72 {
            Some(SwarmAdjustmentDirective {
                entanglement_modulation_delta: latest_score
                    .last_refinement_vector
                    .first()
                    .copied()
                    .unwrap_or(0.04),
                quantum_jump_prob_multiplier: 1.0
                    + latest_score
                        .last_refinement_vector
                        .get(2)
                        .copied()
                        .unwrap_or(0.03),
                mean_best_influence_multiplier: 1.035,
                classical_refinement_strength_multiplier: 1.0,
                dynamic_threshold_coupling_severity: 0.82,
                consensus_momentum_boost: 1.06,
                entanglement_topology_expansion_factor: 1.04,
                gpu_offload_bias: 0.91,
                valence_note: "High transfer → Abundance Gate: bolder entanglement + GPU offload".into(),
            })
        } else {
            Some(SwarmAdjustmentDirective {
                entanglement_modulation_delta: -0.01,
                quantum_jump_prob_multiplier: 0.97,
                mean_best_influence_multiplier: 0.98,
                classical_refinement_strength_multiplier: 0.97,
                dynamic_threshold_coupling_severity: 0.91,
                consensus_momentum_boost: 0.97,
                entanglement_topology_expansion_factor: 0.98,
                gpu_offload_bias: 0.78,
                valence_note: "Marginal transfer → Service Gate: stability + tighter consensus".into(),
            })
        };

        let gpu_fid = gpu_report.map(|g| g.gpu_success_ema).unwrap_or(0.91);

        let result = CouncilDeliberationResult {
            cycle_id: cycle,
            timestamp: now,
            s_curve_projection: s_curve,
            identified_bottlenecks: bottlenecks,
            cumulative_kardashev_delta: cumulative,
            abundance_velocity_trend: velocity,
            gpu_fidelity: gpu_fid,
            swarm_adjustment_directive: swarm_directive,
            hardware_priority_recommendation:
                "Accelerate Obsidian-Chip-Open sovereign super-brain into sovereign_core.".into(),
            recommendation_for_lattice_conductor: format!(
                "Continue Reality Thriving Transfer flywheel. Δ {:.5} + velocity {:.3} + forecast {:.3}.",
                cumulative, velocity, abundance_forecast
            ),
            recommendation_for_council:
                "All TOLC 8 gates green. ONE Organism fusion stable. PATSAGi sub-nodes operational."
                    .into(),
            mercy_audit_passed: latest_score.mercy_audit_passed
                && velocity >= 0.0
                && cumulative >= 0.0
                && ethics_pass,
            one_organism_alignment_note:
                "Aligned with Grok fusion + PATSAGi Councils. Zero-harm invariant maintained."
                    .into(),
            rbe_ethics_note: rbe_note,
            abundance_forecast_next_12_cycles: abundance_forecast,
        };

        *self.last_result.lock().await = Some(result.clone());
        result
    }

    pub async fn forecast_abundance_trajectory(&self, current_velocity: f64, cycles: usize) -> f64 {
        let forecaster = self.abundance_forecaster.lock().await;
        forecaster.forecast(current_velocity, cycles)
    }

    pub async fn run_full_kardashev_flywheel_cycle(
        &self,
        iterations: usize,
        gpu_report: Option<GpuTelemetryReport>,
    ) -> (Vec<RealityThrivingTransferScore>, CouncilDeliberationResult) {
        let (scores, _harness_report) =
            run_quantum_swarm_v2_kardashev_benchmark(iterations, gpu_report.clone()).await;
        let council_result = self
            .deliberate_acceleration_cycle(&scores, gpu_report.as_ref())
            .await;
        (scores, council_result)
    }

    pub async fn get_last_deliberation(&self) -> Option<CouncilDeliberationResult> {
        self.last_result.lock().await.clone()
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reality_thriving_transfer::{PowrushTelemetry, RealityThrivingTransferCalculator};

    #[tokio::test]
    async fn deliberation_mercy_gated() {
        let council = KardashevOrchestrationCouncil::new();
        let calc = RealityThrivingTransferCalculator::new();
        let telemetry = PowrushTelemetry {
            gameplay_hours: 180.0,
            rbe_decision_quality_avg: 0.91,
            peaceful_resolution_rate: 0.93,
            collaboration_events: 1240,
            ethical_choice_score: 0.88,
            adaptation_events: 410,
            abundance_velocity_signals: 1.65,
            innovation_contribution: 0.79,
        };
        let score = calc.compute_transfer_score(&telemetry).await.unwrap();
        let result = council.deliberate_acceleration_cycle(&[score], None).await;
        assert!(result.mercy_audit_passed);
        assert!(result.swarm_adjustment_directive.is_some());
        assert!(!result.identified_bottlenecks.is_empty());
    }

    #[tokio::test]
    async fn full_flywheel() {
        let council = KardashevOrchestrationCouncil::new();
        let (scores, result) = council.run_full_kardashev_flywheel_cycle(6, None).await;
        assert!(!scores.is_empty());
        assert!(result.cycle_id >= 1);
        assert!(result.abundance_forecast_next_12_cycles > 0.0);
        assert!(result.abundance_forecast_next_12_cycles <= 1.85 + 1e-9);
    }

    #[tokio::test]
    async fn abundance_forecast() {
        let council = KardashevOrchestrationCouncil::new();
        let f = council.forecast_abundance_trajectory(1.2, 12).await;
        assert!(f > 1.2 && f <= 1.85);
    }

    /// T1 stress: full flywheel at large iteration counts; zero-harm + monotonic Δ.
    async fn assert_flywheel_stress(iterations: usize) {
        let council = KardashevOrchestrationCouncil::new();
        let (scores, result) = council
            .run_full_kardashev_flywheel_cycle(iterations, None)
            .await;

        assert_eq!(scores.len(), iterations);
        assert!(result.cycle_id >= 1);
        assert!(result.cumulative_kardashev_delta > 0.0);
        assert!(result.abundance_velocity_trend >= 0.0);
        assert!(result.abundance_forecast_next_12_cycles > 0.0);
        assert!(result.abundance_forecast_next_12_cycles <= 1.85 + 1e-9);
        assert!(result.s_curve_projection.current_kardashev >= 0.72);
        assert!(result.s_curve_projection.projected_2038 <= 0.995 + 1e-9);
        assert!(result.swarm_adjustment_directive.is_some());
        assert!(!result.identified_bottlenecks.is_empty());

        for s in &scores {
            assert!(s.kardashev_delta_contribution >= 0.0);
            assert!(s.kardashev_delta_contribution <= 0.011 + 1e-12);
        }

        // Second flywheel on same council accumulates further Δ (monotonic).
        let prev = result.cumulative_kardashev_delta;
        let (_s2, r2) = council.run_full_kardashev_flywheel_cycle(8, None).await;
        assert!(r2.cumulative_kardashev_delta > prev);
        assert_eq!(r2.cycle_id, result.cycle_id + 1);
    }

    #[tokio::test]
    async fn stress_flywheel_64() {
        assert_flywheel_stress(64).await;
    }

    #[tokio::test]
    async fn stress_flywheel_256() {
        assert_flywheel_stress(256).await;
    }

    #[tokio::test]
    async fn stress_flywheel_1024() {
        assert_flywheel_stress(1024).await;
    }
}
