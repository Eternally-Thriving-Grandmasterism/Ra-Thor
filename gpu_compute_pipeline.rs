//! gpu_compute_pipeline.rs — MIGRATION SHIM (v14.9.4)
//!
//! Production public API has moved to the workspace crate:
//!
//!   crates/gpu-compute-pipeline/  (package: `gpu-compute-pipeline` @ 14.9.4)
//!
//! Prefer:
//!
//!   gpu-compute-pipeline = { path = "crates/gpu-compute-pipeline", version = "14.9.4" }
//!   // optional real wgpu:
//!   gpu-compute-pipeline = { path = "crates/gpu-compute-pipeline", features = ["wgpu"] }
//!
//!   use gpu_compute_pipeline::{GpuComputePipeline, GpuTask, GpuTaskResult, create_gpu_pipeline};
//!
//! Crate default = CPU / simulation path (no wgpu dep).
//! Full historical WGSL device path + Live Frame Bridge remains in git history
//! of this file and in monorepo `shaders/`.
//!
//! AG-SML v1.0 — Autonomicity Games Sovereign Mercy License

pub const MIGRATED_TO: &str = "crates/gpu-compute-pipeline";
pub const CRATE_VERSION: &str = "14.9.4";
