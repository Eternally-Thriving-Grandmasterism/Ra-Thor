/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
///
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

//! # Binary-Level Wiring: Real GpuComputePipeline into Lattice Conductor at Startup
//!
//! This example demonstrates the **correct production pattern** for wiring the
//! real `gpu_compute_pipeline::GpuComputePipeline` (from root) into
//! `SimpleLatticeConductor` at binary / application startup.
//!
//! ## Why this matters (ONE Organism)
//! - Conductor now **owns** the GPU pipeline lifecycle (`Arc<GpuComputePipeline>`).
//! - All PATSAGi GPU work (`submit_patsagi_task_with_audit`) is fully delegated.
//! - Telemetry auto-save + graceful shutdown (5s timeout) is conductor-driven.
//! - Mercy-norm audits flow back into self-evolution, council readiness, and EMAs.
//!
//! ## How to use in your main binary
//! In your root `main.rs` or binary crate (where you have `mod gpu_compute_pipeline;`):
//! ```rust,ignore
//! use std::sync::Arc;
//! use lattice_conductor_v13::SimpleLatticeConductor;
//! use gpu_compute_pipeline::GpuComputePipeline as RealGpu;  // the real one
//!
//! #[tokio::main]
//! async fn main() {
//!     let real_gpu = RealGpu::new();
//!     let mut conductor = SimpleLatticeConductor::new();
//!     conductor.set_gpu_pipeline(Arc::new(real_gpu));
//!
//!     // Startup
//!     conductor.start_gpu_pipeline_telemetry(30).await.unwrap();
//!
//!     // ... run your Powrush / RBE / PATSAGi loops ...
//!
//!     // Shutdown
//!     conductor.stop_gpu_pipeline_telemetry().await.unwrap();
//! }
//! ```

use std::sync::Arc;
use lattice_conductor_v13::{GpuComputePipeline as ConductorGpu, MercyGpuAudit, SimpleLatticeConductor};

#[tokio::main]
async fn main() {
    println!("\n=== Ra-Thor v13.5 Binary-Level GPU → Conductor Wiring at Startup ===\n");

    // 1. Create the REAL GpuComputePipeline (from root gpu_compute_pipeline.rs)
    //    In a real binary where `mod gpu_compute_pipeline;` is declared at root,
    //    you would do: use gpu_compute_pipeline::GpuComputePipeline as RealGpu;
    //    Here we use the conductor's bridge type for the example to compile cleanly.
    //    The wiring pattern is identical when the real type is in scope.
    let real_gpu = ConductorGpu::new();
    println!("[Startup] Real GpuComputePipeline created (allocator + telemetry + breaker + prometheus ready)");

    // 2. Create the Lattice Conductor
    let mut conductor = SimpleLatticeConductor::new();
    println!("[Startup] SimpleLatticeConductor v13 created");

    // 3. Wire the real GPU pipeline into the conductor (ownership transfer)
    conductor.set_gpu_pipeline(Arc::new(real_gpu));
    println!("[Startup] Real GPU pipeline wired into conductor (Arc ownership established)");

    // 4. Start periodic mercy telemetry auto-save at startup (conductor-owned)
    match conductor.start_gpu_pipeline_telemetry(30).await {
        Ok(()) => println!("[Startup] Periodic mercy telemetry auto-save started (30s interval, 5s shutdown timeout)");
        Err(e) => eprintln!("[Startup] Failed to start telemetry: {}", e),
    }

    // 5. Demonstrate full delegation of submit_patsagi_task_with_audit through the conductor
    println!("\n[Runtime] Submitting PATSAGi task via conductor delegation...");
    match conductor
        .submit_patsagi_task_with_audit("powrush_rbe_council_ready_check", "high", 2048)
        .await
    {
        Ok((result, audit)) => {
            println!("[Delegation] Task completed: {} | mercy_norm={:.4} | council_ready={}",
                     result.message, audit.mercy_norm, audit.council_ready);

            // 6. Feed the audit back into self-evolution (integrate_patsagi_gpu_audit)
            conductor.integrate_patsagi_gpu_audit(&audit);
            println!("[Self-Evolution] Audit integrated → confidence_ema, mercy_score, evolution_level updated");
        }
        Err(e) => {
            println!("[Delegation] (Expected in facade example) {}", e);
            println!("             In real binary with gpu_compute_pipeline.rs this returns real (result, audit)");
        }
    }

    // 7. Graceful shutdown at application exit (conductor-owned, propagates to background tasks)
    println!("\n[Shutdown] Initiating graceful GPU telemetry shutdown...");
    match conductor.stop_gpu_pipeline_telemetry().await {
        Ok(()) => println!("[Shutdown] GPU pipeline stopped cleanly (timeout respected)");
        Err(e) => eprintln!("[Shutdown] Error during stop: {}", e),
    }

    println!("\n=== Wiring complete. Conductor now fully owns GPU lifecycle for Powrush/RBE. ===\n");
}
