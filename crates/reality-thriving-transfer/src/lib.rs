//! reality-thriving-transfer v14.15.1
//!
//! Reality Thriving Transfer Score + Kardashev benchmark harness.
//! Phase C: Powrush-MMO telemetry contract + offline JSON fixture ingest.
//! Provenance fields optional (Powrush v21.77+).
//!
//! See `POWRUSH_TELEMETRY_CONTRACT.md` and `fixtures/`.
//! AG-SML v1.0 | TOLC 8 Living Mercy Gates | Contact: info@Rathor.ai

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTelemetryReport {
    pub gpu_success_ema: f64,
    pub valence_modulated_offload_score: f64,
    pub dispatch_count: u64,
    pub memory_usage_bytes: u64,
}

impl Default for GpuTelemetryReport {
    fn default() -> Self {
        Self {
            gpu_success_ema: 0.93,
            valence_modulated_offload_score: 0.87,
            dispatch_count: 0,
            memory_usage_bytes: 256 * 1024 * 1024,
        }
    }
}

/// Canonical Powrush → Ra-Thor telemetry (schema powrush_telemetry_v1).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PowrushTelemetry {
    pub gameplay_hours: f64,
    pub rbe_decision_quality_avg: f64,
    pub peaceful_resolution_rate: f64,
    pub collaboration_events: u64,
    pub ethical_choice_score: f64,
    pub adaptation_events: u64,
    pub abundance_velocity_signals: f64,
    pub innovation_contribution: f64,
}

/// Single-session JSON envelope from Powrush-MMO exporters / fixtures.
/// Optional provenance fields (v21.77+) for session-keyed council feedback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowrushTelemetryEnvelope {
    pub schema: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub exported_at_unix: Option<u64>,
    #[serde(default)]
    pub export_seq: Option<u64>,
    pub telemetry: PowrushTelemetry,
}

/// One entry inside a batch envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowrushTelemetrySession {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub exported_at_unix: Option<u64>,
    pub telemetry: PowrushTelemetry,
}

/// Batch JSON envelope (schema powrush_telemetry_batch_v1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowrushTelemetryBatch {
    pub schema: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub exported_at_unix: Option<u64>,
    pub sessions: Vec<PowrushTelemetrySession>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealityThrivingTransferScore {
    pub raw_transfer_score: f64,
    pub mercy_valence_adjusted: f64,
    pub ema_refined_transfer: f64,
    pub confidence: f64,
    pub kardashev_delta_contribution: f64,
    pub abundance_velocity_index: f64,
    pub ethics_collaboration_index: f64,
    pub last_refinement_vector: Vec<f64>,
    pub mercy_audit_passed: bool,
    pub timestamp: u64,
    pub council_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KardashevOrchestrationReport {
    pub cumulative_kardashev_delta: f64,
    pub abundance_velocity_trend: f64,
    pub transfer_score_trend_ema: f64,
    pub swarm_convergence_improvement: f64,
    pub gpu_fidelity: f64,
    pub mercy_gates_status: String,
    pub recommendation_for_council: String,
}

// =============================================================================
// Fixture / exporter ingest
// =============================================================================

/// Parse a single-session `powrush_telemetry_v1` JSON string.
pub fn parse_powrush_telemetry_json(json: &str) -> Result<PowrushTelemetryEnvelope, String> {
    let env: PowrushTelemetryEnvelope = serde_json::from_str(json)
        .map_err(|e| format!("Mercy Gate (Truth): invalid powrush_telemetry_v1 JSON: {}", e))?;
    if env.schema != "powrush_telemetry_v1" {
        return Err(format!(
            "Mercy Gate (Truth): expected schema powrush_telemetry_v1, got '{}'",
            env.schema
        ));
    }
    Ok(env)
}

/// Parse a batch `powrush_telemetry_batch_v1` JSON string.
pub fn parse_powrush_telemetry_batch_json(json: &str) -> Result<PowrushTelemetryBatch, String> {
    let batch: PowrushTelemetryBatch = serde_json::from_str(json)
        .map_err(|e| format!("Mercy Gate (Truth): invalid powrush_telemetry_batch_v1 JSON: {}", e))?;
    if batch.schema != "powrush_telemetry_batch_v1" {
        return Err(format!(
            "Mercy Gate (Truth): expected schema powrush_telemetry_batch_v1, got '{}'",
            batch.schema
        ));
    }
    if batch.sessions.is_empty() {
        return Err("Mercy Gate (Truth): batch contains no sessions".into());
    }
    Ok(batch)
}

/// Score every session in a batch sequentially (shared calculator EMAs).
pub async fn compute_scores_from_batch(
    calc: &RealityThrivingTransferCalculator,
    batch: &PowrushTelemetryBatch,
) -> Result<Vec<(String, RealityThrivingTransferScore)>, String> {
    let mut out = Vec::with_capacity(batch.sessions.len());
    for session in &batch.sessions {
        let score = calc.compute_transfer_score(&session.telemetry).await?;
        out.push((session.label.clone(), score));
    }
    Ok(out)
}

pub struct RealityThrivingTransferCalculator {
    transfer_ema: Arc<Mutex<f64>>,
    valence_ema: Arc<Mutex<f64>>,
    velocity_ema: Arc<Mutex<f64>>,
    confidence_ema: Arc<Mutex<f64>>,
    total_transfers: Arc<Mutex<u64>>,
    last_update: Arc<Mutex<Instant>>,
}

impl Default for RealityThrivingTransferCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl RealityThrivingTransferCalculator {
    pub fn new() -> Self {
        Self {
            transfer_ema: Arc::new(Mutex::new(0.42)),
            valence_ema: Arc::new(Mutex::new(0.87)),
            velocity_ema: Arc::new(Mutex::new(0.31)),
            confidence_ema: Arc::new(Mutex::new(0.78)),
            total_transfers: Arc::new(Mutex::new(0)),
            last_update: Arc::new(Mutex::new(Instant::now())),
        }
    }

    pub async fn compute_transfer_score(
        &self,
        telemetry: &PowrushTelemetry,
    ) -> Result<RealityThrivingTransferScore, String> {
        if !(0.0..=1.0).contains(&telemetry.rbe_decision_quality_avg) {
            return Err("Mercy Gate (Truth): rbe_decision_quality_avg out of [0,1] bounds".into());
        }
        if !(0.0..=1.0).contains(&telemetry.ethical_choice_score) {
            return Err("Mercy Gate (Truth): ethical_choice_score out of bounds".into());
        }
        if telemetry.abundance_velocity_signals < 0.0 {
            return Err("Mercy Gate (Abundance): Negative abundance signals rejected — zero-harm".into());
        }

        let total_weight = 0.28 + 0.22 + 0.18 + 0.15 + 0.12 + 0.05;
        let raw = (telemetry.rbe_decision_quality_avg * 0.28
            + telemetry.peaceful_resolution_rate * 0.22
            + (telemetry.collaboration_events as f64).min(500.0) / 500.0 * 0.18
            + telemetry.ethical_choice_score * 0.15
            + telemetry.adaptation_events.min(300) as f64 / 300.0 * 0.12
            + telemetry.innovation_contribution * 0.05)
            / total_weight;

        let mercy_adjusted = if raw >= 0.68 {
            (raw * 1.08).min(0.995)
        } else if raw >= 0.42 {
            raw * 1.03
        } else {
            raw * 0.82
        };

        let alpha = 0.28;
        {
            let mut ema = self.transfer_ema.lock().await;
            *ema = alpha * mercy_adjusted + (1.0 - alpha) * *ema;
        }
        {
            let mut v_ema = self.valence_ema.lock().await;
            let valence_signal = (mercy_adjusted + telemetry.ethical_choice_score) / 2.0;
            *v_ema = 0.22 * valence_signal + (1.0 - 0.22) * *v_ema;
        }
        {
            let mut vel_ema = self.velocity_ema.lock().await;
            *vel_ema = 0.19 * telemetry.abundance_velocity_signals.min(1.8) + (1.0 - 0.19) * *vel_ema;
        }

        let ema_refined = *self.transfer_ema.lock().await;
        let valence = *self.valence_ema.lock().await;

        let attempts = *self.total_transfers.lock().await;
        let base_conf = if attempts < 6 {
            0.81
        } else {
            (ema_refined * 0.65 + valence * 0.35).clamp(0.71, 0.985)
        };
        {
            let mut c_ema = self.confidence_ema.lock().await;
            *c_ema = 0.31 * base_conf + (1.0 - 0.31) * *c_ema;
        }
        let confidence = *self.confidence_ema.lock().await;

        let kardashev_delta =
            (mercy_adjusted * 0.0095 + telemetry.abundance_velocity_signals * 0.0028).min(0.011);

        let refinement = vec![
            (mercy_adjusted - 0.5) * 0.18,
            (0.5 - mercy_adjusted) * 0.09,
            mercy_adjusted * 0.07,
        ];

        *self.total_transfers.lock().await += 1;
        *self.last_update.lock().await = Instant::now();

        let mercy_audit_passed =
            mercy_adjusted >= 0.0 && confidence >= 0.71 && kardashev_delta >= 0.0;

        Ok(RealityThrivingTransferScore {
            raw_transfer_score: raw,
            mercy_valence_adjusted: mercy_adjusted,
            ema_refined_transfer: ema_refined,
            confidence,
            kardashev_delta_contribution: kardashev_delta,
            abundance_velocity_index: *self.velocity_ema.lock().await,
            ethics_collaboration_index: (telemetry.ethical_choice_score
                + telemetry.peaceful_resolution_rate)
                / 2.0,
            last_refinement_vector: refinement,
            mercy_audit_passed,
            timestamp: now_secs(),
            council_note: format!(
                "Transfer {} | Valence {:.3} | Kardashev Δ {:.5} | Mercy gates: {}",
                if mercy_audit_passed { "PASSED" } else { "DAMPENED" },
                valence,
                kardashev_delta,
                if mercy_audit_passed { "all green" } else { "compassion engaged" }
            ),
        })
    }

    pub async fn apply_transfer_feedback_to_swarm(&self, score: &RealityThrivingTransferScore) {
        let _ = (
            score.last_refinement_vector.first().copied().unwrap_or(0.0),
            score.last_refinement_vector.get(1).copied().unwrap_or(0.0),
            score.last_refinement_vector.get(2).copied().unwrap_or(0.0),
        );
    }

    pub async fn get_current_valence(&self) -> f64 {
        *self.valence_ema.lock().await
    }
}

pub async fn run_quantum_swarm_v2_kardashev_benchmark(
    iterations: usize,
    gpu_report: Option<GpuTelemetryReport>,
) -> (Vec<RealityThrivingTransferScore>, KardashevOrchestrationReport) {
    let calculator = RealityThrivingTransferCalculator::new();
    let mut scores = Vec::with_capacity(iterations);
    let mut cumulative_kardashev: f64 = 0.0;

    for i in 0..iterations {
        let progress = (i as f64 / iterations.max(1) as f64).min(0.97);
        let telemetry = PowrushTelemetry {
            gameplay_hours: 12.0 + i as f64 * 0.8,
            rbe_decision_quality_avg: 0.61 + progress * 0.31,
            peaceful_resolution_rate: 0.67 + progress * 0.27,
            collaboration_events: 180 + (i as u64 * 7),
            ethical_choice_score: 0.58 + progress * 0.34,
            adaptation_events: 90 + (i as u64 * 4),
            abundance_velocity_signals: 0.9 + progress * 0.85,
            innovation_contribution: 0.55 + progress * 0.38,
        };

        match calculator.compute_transfer_score(&telemetry).await {
            Ok(score) => {
                cumulative_kardashev += score.kardashev_delta_contribution;
                scores.push(score);
            }
            Err(e) => eprintln!("Mercy Gate engaged on iteration {}: {}", i, e),
        }
    }

    let final_score = scores.last().cloned().unwrap_or_else(|| RealityThrivingTransferScore {
        raw_transfer_score: 0.0,
        mercy_valence_adjusted: 0.0,
        ema_refined_transfer: 0.0,
        confidence: 0.5,
        kardashev_delta_contribution: 0.0,
        abundance_velocity_index: 0.0,
        ethics_collaboration_index: 0.0,
        last_refinement_vector: vec![0.0; 3],
        mercy_audit_passed: false,
        timestamp: 0,
        council_note: "No valid transfers".into(),
    });

    let gpu_fid = gpu_report
        .as_ref()
        .map(|g| g.gpu_success_ema)
        .unwrap_or(0.94);

    let report = KardashevOrchestrationReport {
        cumulative_kardashev_delta: cumulative_kardashev,
        abundance_velocity_trend: final_score.abundance_velocity_index,
        transfer_score_trend_ema: final_score.ema_refined_transfer,
        swarm_convergence_improvement: (final_score.mercy_valence_adjusted - 0.5) * 0.22,
        gpu_fidelity: gpu_fid,
        mercy_gates_status: if final_score.mercy_audit_passed {
            "All TOLC 8 + Mercy Gates PASSED".into()
        } else {
            "Compassion gate engaged — zero harm maintained".into()
        },
        recommendation_for_council:
            "Continue flywheel. Current transfer velocity supports accelerated Kardashev climb."
                .into(),
    };

    (scores, report)
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

    const FIXTURE_HIGH: &str = include_str!("../fixtures/session_high_mercy.json");
    const FIXTURE_MARGINAL: &str = include_str!("../fixtures/session_marginal.json");
    const FIXTURE_EARLY: &str = include_str!("../fixtures/session_early_player.json");
    const FIXTURE_BATCH: &str = include_str!("../fixtures/batch_three_sessions.json");

    #[tokio::test]
    async fn mercy_gate_rejects_invalid() {
        let calc = RealityThrivingTransferCalculator::new();
        let bad = PowrushTelemetry {
            gameplay_hours: 10.0,
            rbe_decision_quality_avg: 1.3,
            peaceful_resolution_rate: 0.8,
            collaboration_events: 50,
            ethical_choice_score: 0.7,
            adaptation_events: 20,
            abundance_velocity_signals: 1.1,
            innovation_contribution: 0.6,
        };
        assert!(calc.compute_transfer_score(&bad).await.is_err());
    }

    #[tokio::test]
    async fn positive_transfer() {
        let calc = RealityThrivingTransferCalculator::new();
        let good = PowrushTelemetry {
            gameplay_hours: 47.0,
            rbe_decision_quality_avg: 0.89,
            peaceful_resolution_rate: 0.91,
            collaboration_events: 312,
            ethical_choice_score: 0.87,
            adaptation_events: 148,
            abundance_velocity_signals: 1.6,
            innovation_contribution: 0.82,
        };
        let score = calc.compute_transfer_score(&good).await.unwrap();
        assert!(score.mercy_audit_passed);
        assert!(score.kardashev_delta_contribution > 0.004);
        assert!(score.kardashev_delta_contribution <= 0.011);
    }

    #[tokio::test]
    async fn benchmark_runs() {
        let (scores, report) = run_quantum_swarm_v2_kardashev_benchmark(8, None).await;
        assert!(!scores.is_empty());
        assert!(report.cumulative_kardashev_delta > 0.0);
    }

    async fn assert_stress_bounds(iterations: usize) {
        let (scores, report) = run_quantum_swarm_v2_kardashev_benchmark(iterations, None).await;
        assert_eq!(scores.len(), iterations);
        assert!(report.cumulative_kardashev_delta > 0.0);
        let mut running = 0.0;
        for s in &scores {
            assert!(s.kardashev_delta_contribution >= 0.0);
            assert!(s.kardashev_delta_contribution <= 0.011 + 1e-12);
            assert!(s.abundance_velocity_index >= 0.0);
            assert!(s.abundance_velocity_index <= 1.85 + 1e-9);
            assert!(s.mercy_valence_adjusted <= 0.995 + 1e-9);
            assert!(s.confidence >= 0.71 - 1e-9);
            running += s.kardashev_delta_contribution;
        }
        assert!((running - report.cumulative_kardashev_delta).abs() < 1e-9);
    }

    #[tokio::test]
    async fn stress_benchmark_64() {
        assert_stress_bounds(64).await;
    }

    #[tokio::test]
    async fn stress_benchmark_256() {
        assert_stress_bounds(256).await;
    }

    #[tokio::test]
    async fn stress_benchmark_1024() {
        assert_stress_bounds(1024).await;
    }

    #[test]
    fn parse_high_mercy_fixture() {
        let env = parse_powrush_telemetry_json(FIXTURE_HIGH).unwrap();
        assert_eq!(env.label, "high_mercy_council_session");
        assert!(env.telemetry.rbe_decision_quality_avg > 0.9);
        assert!(env.telemetry.collaboration_events >= 400);
        // Provenance optional — fixtures may omit
        assert!(env.session_id.is_none() || env.session_id.as_ref().map(|s| !s.is_empty()).unwrap_or(false));
    }

    #[test]
    fn parse_marginal_and_early_fixtures() {
        let m = parse_powrush_telemetry_json(FIXTURE_MARGINAL).unwrap();
        assert!(m.telemetry.ethical_choice_score < 0.5);
        let e = parse_powrush_telemetry_json(FIXTURE_EARLY).unwrap();
        assert!(e.telemetry.gameplay_hours < 5.0);
    }

    #[test]
    fn parse_batch_fixture() {
        let batch = parse_powrush_telemetry_batch_json(FIXTURE_BATCH).unwrap();
        assert_eq!(batch.sessions.len(), 3);
        assert_eq!(batch.schema, "powrush_telemetry_batch_v1");
    }

    #[test]
    fn parse_with_provenance() {
        let json = r#"{
          "schema": "powrush_telemetry_v1",
          "source": "powrush-mmo-server",
          "label": "server_live_session",
          "session_id": "server_live_session_1721600000",
          "exported_at_unix": 1721600000,
          "export_seq": 3,
          "telemetry": {
            "gameplay_hours": 1.0,
            "rbe_decision_quality_avg": 0.8,
            "peaceful_resolution_rate": 0.9,
            "collaboration_events": 10,
            "ethical_choice_score": 0.85,
            "adaptation_events": 5,
            "abundance_velocity_signals": 1.1,
            "innovation_contribution": 0.4
          }
        }"#;
        let env = parse_powrush_telemetry_json(json).unwrap();
        assert_eq!(env.session_id.as_deref(), Some("server_live_session_1721600000"));
        assert_eq!(env.exported_at_unix, Some(1721600000));
        assert_eq!(env.export_seq, Some(3));
    }

    #[tokio::test]
    async fn score_high_mercy_fixture() {
        let env = parse_powrush_telemetry_json(FIXTURE_HIGH).unwrap();
        let calc = RealityThrivingTransferCalculator::new();
        let score = calc.compute_transfer_score(&env.telemetry).await.unwrap();
        assert!(score.mercy_audit_passed);
        assert!(score.kardashev_delta_contribution > 0.004);
        assert!(score.kardashev_delta_contribution <= 0.011);
        assert!(score.ethics_collaboration_index >= 0.68);
    }

    #[tokio::test]
    async fn score_batch_fixture_all_sessions() {
        let batch = parse_powrush_telemetry_batch_json(FIXTURE_BATCH).unwrap();
        let calc = RealityThrivingTransferCalculator::new();
        let scored = compute_scores_from_batch(&calc, &batch).await.unwrap();
        assert_eq!(scored.len(), 3);
        for (label, score) in &scored {
            assert!(!label.is_empty());
            assert!(score.kardashev_delta_contribution >= 0.0);
            assert!(score.kardashev_delta_contribution <= 0.011 + 1e-12);
            assert!(score.abundance_velocity_index >= 0.0);
        }
        let high = scored.iter().find(|(l, _)| l.contains("high_mercy")).unwrap();
        let marginal = scored.iter().find(|(l, _)| l.contains("marginal")).unwrap();
        assert!(high.1.raw_transfer_score > marginal.1.raw_transfer_score);
    }

    #[test]
    fn reject_wrong_schema() {
        let bad = r#"{"schema":"nope","telemetry":{"gameplay_hours":1.0,"rbe_decision_quality_avg":0.5,"peaceful_resolution_rate":0.5,"collaboration_events":1,"ethical_choice_score":0.5,"adaptation_events":1,"abundance_velocity_signals":0.5,"innovation_contribution":0.5}}"#;
        assert!(parse_powrush_telemetry_json(bad).is_err());
    }
}
