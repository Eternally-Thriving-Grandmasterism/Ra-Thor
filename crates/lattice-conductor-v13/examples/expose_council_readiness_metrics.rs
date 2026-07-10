/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
///
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

//! # Expose Council Readiness Metrics from the Manager
//!
//! This example adds **CouncilReadinessMetrics** exposure on `PowrushLifecycleManager`.
//!
//! The metrics struct surfaces the key PATSAGi / self-evolution signals:
//! - `council_ready: bool`
//! - `mercy_norm: f64`
//! - `suggested_confidence_delta: f64`
//! - `evolution_level: u32`
//! - `last_updated: u64` (tick count)
//!
//! These metrics can be consumed by:
//! - PATSAGi Councils for governance decisions
//! - Self-evolution hooks
//! - Prometheus / observability layers
//! - Powrush RBE simulation dashboards

use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::time::{sleep, Duration};
use lattice_conductor_v13::{GpuComputePipeline as ConductorGpu, MercyGpuAudit, SimpleLatticeConductor};

/// Public metrics struct for council readiness and self-evolution state
#[derive(Debug, Clone)]
pub struct CouncilReadinessMetrics {
    pub council_ready: bool,
    pub mercy_norm: f64,
    pub suggested_confidence_delta: f64,
    pub evolution_level: u32,
    pub last_updated_tick: u64,
}

pub struct PowrushLifecycleManager {
    conductor: SimpleLatticeConductor,
    gpu: Arc<ConductorGpu>,
    shutdown_tx: broadcast::Sender<()>,
    tick_count: u64,
    last_audit: Option<MercyGpuAudit>,
}

impl PowrushLifecycleManager {
    pub fn new() -> Self {
        let gpu = Arc::new(ConductorGpu::new());
        let mut conductor = SimpleLatticeConductor::new();
        conductor.set_gpu_pipeline(gpu.clone());

        let (shutdown_tx, _) = broadcast::channel(16);

        Self {
            conductor,
            gpu,
            shutdown_tx,
            tick_count: 0,
            last_audit: None,
        }
    }

    /// Expose current council readiness metrics (for PATSAGi councils, self-evolution, observability)
    pub fn get_council_readiness_metrics(&self) -> CouncilReadinessMetrics {
        if let Some(audit) = &self.last_audit {
            CouncilReadinessMetrics {
                council_ready: audit.council_ready,
                mercy_norm: audit.mercy_norm,
                suggested_confidence_delta: audit.suggested_confidence_delta,
                evolution_level: self.conductor.evolution_level(),
                last_updated_tick: self.tick_count,
            }
        } else {
            // Default / initial state
            CouncilReadinessMetrics {
                council_ready: false,
                mercy_norm: 0.0,
                suggested_confidence_delta: 0.0,
                evolution_level: self.conductor.evolution_level(),
                last_updated_tick: self.tick_count,
            }
        }
    }

    pub async fn start(&mut self) {
        println!("[Manager] Starting GPU telemetry + Powrush RBE simulation with metrics exposure...");
        let _ = self.conductor.start_gpu_pipeline_telemetry(30).await;

        let conductor = self.conductor.clone_for_background();
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        // We need interior mutability for tick_count + last_audit in the background task.
        // For simplicity in this example we keep state updates in the main thread after each tick.
        // In production you would use Arc<Mutex<...>> or channels.

        tokio::spawn(async move {
            println!("[RBE Loop] Powrush RBE simulation loop with council metrics started");
            loop {
                tokio::select! {
                    _ = sleep(Duration::from_secs(5)) => {
                        if let Ok((result, audit)) = conductor
                            .submit_patsagi_task_with_audit(
                                "powrush_rbe_council_readiness_tick",
                                "medium",
                                4096,
                            )
                            .await
                        {
                            conductor.integrate_patsagi_gpu_audit(&audit);
                            println!(
                                "[RBE Loop] Tick | mercy_norm={:.4} | council_ready={} | confidence_delta={:+.4}",
                                audit.mercy_norm, audit.council_ready, audit.suggested_confidence_delta
                            );
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        println!("[RBE Loop] Shutdown signal — stopping RBE loop");
                        break;
                    }
                }
            }
        });
    }

    pub async fn submit_patsagi_gpu_task(
        &self,
        query: &str,
        intensity: &str,
        buffer_size: usize,
    ) -> Result<(lattice_conductor_v13::GpuTaskResult, MercyGpuAudit), String> {
        self.conductor
            .submit_patsagi_task_with_audit(query, intensity, buffer_size)
            .await
    }

    pub fn integrate_gpu_audit(&mut self, audit: &MercyGpuAudit) {
        self.last_audit = Some(audit.clone());
        self.tick_count += 1;
        self.conductor.integrate_patsagi_gpu_audit(audit);
    }

    pub async fn stop(&mut self) {
        println!("[Manager] Graceful shutdown...");
        let _ = self.shutdown_tx.send(());
        let _ = self.conductor.stop_gpu_pipeline_telemetry().await;
        sleep(Duration::from_millis(200)).await;
        println!("[Manager] All loops stopped cleanly");
    }
}

#[tokio::main]
async fn main() {
    println!("\n=== Ra-Thor v13.9 — Expose Council Readiness Metrics ===\n");

    let mut manager = PowrushLifecycleManager::new();

    manager.start().await;

    // Demonstrate metrics exposure after some ticks
    for i in 0..3 {
        sleep(Duration::from_secs(6)).await;

        let metrics = manager.get_council_readiness_metrics();
        println!(
            "[Main] Council Metrics @ tick {}: ready={} | mercy_norm={:.4} | confidence_delta={:+.4} | evolution_level={}",
            metrics.last_updated_tick,
            metrics.council_ready,
            metrics.mercy_norm,
            metrics.suggested_confidence_delta,
            metrics.evolution_level
        );
    }

    manager.stop().await;

    println!("\n=== Metrics exposure complete. Ready for PATSAGi council consumption. ===\n");
}
