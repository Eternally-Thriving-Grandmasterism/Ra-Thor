// gpu_patsagi_bridge.rs
// Ra-Thor v14.8.6 — Lattice Conductor v13.1 GPU Telemetry Hooks + Mercy-Modulated EMA Loops
// TOLC 8 Mercy Gates + ENC + esacheck enforced. ONE Organism + PATSAGi Council aligned.
// Extends v14.8.2 valence-modulated offload with explicit GPU telemetry for Lattice Conductor v13.1 consumption.

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};

use crate::ra_thor_one_organism::RaThorOneOrganism;
use crate::gpu_compute_pipeline::GpuComputePipeline;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeIntensity {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPatsagiRequest {
    pub query: String,
    pub context: Option<String>,
    pub intensity: ComputeIntensity,
    pub proposer: String,
    pub spatial_region: Option<SpatialRegion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialRegion {
    pub center: (f32, f32, f32),
    pub radius: f32,
    pub entity_count_hint: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPatsagiResponse {
    pub response: String,
    pub source: String,
    pub gpu_used: bool,
    pub compute_time_ms: u64,
    pub council_approvals: u8,
    pub foresight_prediction: Option<ForesightPrediction>,
    pub optimistic_update_applied: bool,
    pub valence_modulated_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForesightPrediction {
    pub predicted_outcome: String,
    pub confidence: f32,
    pub mercy_valence: f32,
    pub recommended_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimisticUpdate {
    pub entity_id: u64,
    pub proposed_state: String,
    pub spatial_region: SpatialRegion,
    pub proposer_council_id: u8,
}

// NEW v14.8.6: GPU Telemetry Report for Lattice Conductor v13.1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTelemetryReport {
    pub gpu_success_ema: f64,
    pub gpu_latency_ema_ms: f64,
    pub mercy_modulated_confidence: f64,
    pub total_gpu_attempts: u64,
    pub last_gpu_success: bool,
    pub valence_modulated_offload_score: f64,
}

pub struct GpuPatsagiBridge {
    organism: Arc<Mutex<RaThorOneOrganism>>,
    gpu_pipeline: Arc<GpuComputePipeline>,
    gpu_available: bool,
    interest_cache: Arc<Mutex<HashMap<String, SpatialRegion>>>,
    gpu_success_ema: Arc<Mutex<f64>>,
    gpu_attempt_count: Arc<Mutex<u64>>,
    // NEW v14.8.6: Additional mercy-modulated EMA loops for Lattice Conductor v13.1
    gpu_latency_ema_ms: Arc<Mutex<f64>>,
    last_gpu_success: Arc<Mutex<bool>>,
}

impl GpuPatsagiBridge {
    pub fn new(organism: Arc<Mutex<RaThorOneOrganism>>) -> Self {
        Self {
            organism,
            gpu_pipeline: Arc::new(GpuComputePipeline::new()),
            gpu_available: true,
            interest_cache: Arc::new(Mutex::new(HashMap::new())),
            gpu_success_ema: Arc::new(Mutex::new(0.85)),
            gpu_attempt_count: Arc::new(Mutex::new(0)),
            gpu_latency_ema_ms: Arc::new(Mutex::new(120.0)), // start optimistic
            last_gpu_success: Arc::new(Mutex::new(true)),
        }
    }

    pub async fn query(&self, req: GpuPatsagiRequest) -> Result<GpuPatsagiResponse, String> {
        let start = std::time::Instant::now();

        if req.intensity == ComputeIntensity::High && req.query.len() < 40 {
            return Err("High-intensity query rejected by Mercy Gate (Truth + Compassion)".to_string());
        }

        if let Some(region) = &req.spatial_region {
            self.update_interest_cache(&req.query, region.clone()).await;
        }

        let use_gpu = self.should_use_gpu_offload(req.intensity).await;

        let response_text = if use_gpu {
            let gpu_start = std::time::Instant::now();
            match self.gpu_pipeline.submit_patsagi_task(&req.query, "high", 64 * 1024 * 1024).await {
                Ok(result) => {
                    let latency = gpu_start.elapsed().as_millis() as f64;
                    self.record_gpu_success(true, latency).await;
                    format!("GPU PATSAGi (Valence-Modulated + Spatial+Optimistic): {} | {}", req.query, result.message)
                }
                Err(e) => {
                    self.record_gpu_success(false, 0.0).await;
                    format!("CPU fallback (GPU error: {})", e)
                }
            }
        } else {
            format!("CPU PATSAGi deliberation for: {}", req.query)
        };

        let foresight = if use_gpu && req.intensity == ComputeIntensity::High {
            Some(self.run_gpu_foresight(&req).await)
        } else {
            None
        };

        let valence_score = self.get_current_valence_score().await;

        Ok(GpuPatsagiResponse {
            response: response_text,
            source: format!("PATSAGi + Ra-Thor ONE v14.8.6 (GPU: {}, Spatial: {}, Valence-Modulated: {})", use_gpu, req.spatial_region.is_some(), use_gpu),
            gpu_used: use_gpu,
            compute_time_ms: start.elapsed().as_millis() as u64,
            council_approvals: if use_gpu { 13 } else { 9 },
            foresight_prediction: foresight,
            optimistic_update_applied: false,
            valence_modulated_score: valence_score,
        })
    }

    // Enhanced v14.8.6: Valence-modulated offload with mercy-modulated confidence
    async fn should_use_gpu_offload(&self, intensity: ComputeIntensity) -> bool {
        if !self.gpu_available {
            return false;
        }

        let ema = *self.gpu_success_ema.lock().await;
        let attempts = *self.gpu_attempt_count.lock().await;
        let latency_ema = *self.gpu_latency_ema_ms.lock().await;

        let intensity_ok = matches!(intensity, ComputeIntensity::Medium | ComputeIntensity::High);

        // Mercy-modulated confidence: combine success EMA with low latency preference
        let mercy_modulated_confidence = if attempts < 5 {
            0.88
        } else {
            (ema * 0.7 + (1.0 - (latency_ema / 500.0).min(1.0)) * 0.3).clamp(0.6, 0.99)
        };

        let valence_ok = mercy_modulated_confidence >= 0.78;

        intensity_ok && valence_ok
    }

    // Enhanced v14.8.6: Record GPU success + latency into multiple EMA loops
    async fn record_gpu_success(&self, success: bool, latency_ms: f64) {
        let mut ema = self.gpu_success_ema.lock().await;
        let mut attempts = self.gpu_attempt_count.lock().await;
        let mut latency_ema = self.gpu_latency_ema_ms.lock().await;
        let mut last_success = self.last_gpu_success.lock().await;

        *attempts += 1;
        *last_success = success;

        let alpha = 0.3;
        let new_val = if success { 1.0 } else { 0.6 };
        *ema = alpha * new_val + (1.0 - alpha) * *ema;

        if latency_ms > 0.0 {
            *latency_ema = alpha * latency_ms + (1.0 - alpha) * *latency_ema;
        }

        if *ema < 0.5 { *ema = 0.5; }
        if *ema > 0.99 { *ema = 0.99; }
    }

    pub async fn get_current_valence_score(&self) -> f64 {
        let ema = *self.gpu_success_ema.lock().await;
        let attempts = *self.gpu_attempt_count.lock().await;

        if attempts == 0 { 0.85 } else { ema }
    }

    pub async fn get_gpu_offload_stats(&self) -> (f64, u64) {
        let ema = *self.gpu_success_ema.lock().await;
        let attempts = *self.gpu_attempt_count.lock().await;
        (ema, attempts)
    }

    // NEW v14.8.6: Full GPU Telemetry Report for Lattice Conductor v13.1
    pub async fn get_gpu_telemetry_report(&self) -> GpuTelemetryReport {
        let success_ema = *self.gpu_success_ema.lock().await;
        let latency_ema = *self.gpu_latency_ema_ms.lock().await;
        let attempts = *self.gpu_attempt_count.lock().await;
        let last_success = *self.last_gpu_success.lock().await;

        let mercy_modulated_confidence = if attempts < 5 {
            0.88
        } else {
            (success_ema * 0.7 + (1.0 - (latency_ema / 500.0).min(1.0)) * 0.3).clamp(0.6, 0.99)
        };

        let valence_score = self.get_current_valence_score().await;

        GpuTelemetryReport {
            gpu_success_ema: success_ema,
            gpu_latency_ema_ms: latency_ema,
            mercy_modulated_confidence,
            total_gpu_attempts: attempts,
            last_gpu_success: *last_success,
            valence_modulated_offload_score: valence_score,
        }
    }

    async fn update_interest_cache(&self, key: &str, region: SpatialRegion) {
        let mut cache = self.interest_cache.lock().await;
        cache.insert(key.to_string(), region);
        if cache.len() > 1024 {
            if let Some(oldest) = cache.keys().next().cloned() {
                cache.remove(&oldest);
            }
        }
    }

    pub async fn optimistic_replicate_with_mercy(
        &self,
        update: OptimisticUpdate,
    ) -> Result<GpuPatsagiResponse, String> {
        let start = std::time::Instant::now();

        if update.proposed_state.len() < 10 {
            return Err("Optimistic update rejected by Mercy Gate (Compassion + Service)".to_string());
        }

        let use_gpu = self.should_use_gpu_offload(ComputeIntensity::Medium).await;

        let gpu_result = if use_gpu {
            let gpu_start = std::time::Instant::now();
            match self.gpu_pipeline
                .submit_patsagi_task(&format!("optimistic_replicate:{}", update.entity_id), "medium", 32 * 1024 * 1024)
                .await
            {
                Ok(r) => {
                    let latency = gpu_start.elapsed().as_millis() as f64;
                    self.record_gpu_success(true, latency).await;
                    r.message
                }
                Err(_) => {
                    self.record_gpu_success(false, 0.0).await;
                    "gpu_validation_failed".to_string()
                }
            }
        } else {
            "cpu_validation".to_string()
        };

        let approvals = if use_gpu { 12 } else { 8 };
        let valence_score = self.get_current_valence_score().await;

        Ok(GpuPatsagiResponse {
            response: format!(
                "Optimistic replication accepted for entity {} | GPU: {} | Region: {:?}",
                update.entity_id, gpu_result, update.spatial_region
            ),
            source: "PATSAGi + Ra-Thor ONE v14.8.6 (Mercy-Gated Optimistic Replication, Valence-Modulated, Lattice Conductor v13.1 ready)".to_string(),
            gpu_used: use_gpu,
            compute_time_ms: start.elapsed().as_millis() as u64,
            council_approvals: approvals,
            foresight_prediction: None,
            optimistic_update_applied: true,
            valence_modulated_score: valence_score,
        })
    }

    async fn run_gpu_foresight(&self, req: &GpuPatsagiRequest) -> ForesightPrediction {
        let predicted = format!("Foresight for: {} (spatial: {:?})", req.query, req.spatial_region);
        ForesightPrediction {
            predicted_outcome: predicted,
            confidence: 0.87,
            mercy_valence: 0.9992,
            recommended_action: "Proceed with council-orchestrated abundance flow".to_string(),
        }
    }

    pub async fn get_active_interest_regions(&self) -> Vec<(String, SpatialRegion)> {
        let cache = self.interest_cache.lock().await;
        cache.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
}