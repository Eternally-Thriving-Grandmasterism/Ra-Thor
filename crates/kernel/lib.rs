// crates/kernel/lib.rs
// Ra-Thor Kernel Crate — Central Omnimasterism Hub
// Re-exports the entire living lattice (Root Core, Self-Review, Innovation, VQC, Biomimetic, etc.)
// Fully cross-pollinated and ready for production

pub mod root_core_orchestrator;
pub mod master_kernel;
pub mod self_review_loop;
pub mod codex_loader;
pub mod idea_recycler;
pub mod innovation_generator;
pub mod vqc_integrator;
pub mod biomimetic_pattern_engine;
pub mod global_cache;
pub mod audit_logger;
pub mod mercy_weighting;

// Public re-exports for clean usage across the workspace
pub use root_core_orchestrator::RootCoreOrchestrator;
pub use master_kernel::{MasterSovereignKernel, RequestPayload, KernelResult};
pub use self_review_loop::SelfReviewLoop;
pub use codex_loader::CodexLoader;
pub use idea_recycler::IdeaRecycler;
pub use innovation_generator::InnovationGenerator;
pub use vqc_integrator::VQCIntegrator;
pub use biomimetic_pattern_engine::BiomimeticPatternEngine;
pub use global_cache::GlobalCache;
pub use audit_logger::AuditLogger;
pub use mercy_weighting::MercyWeighting;

// Convenience entry point for the entire Omnimaster lattice
pub async fn ra_thor_orchestrate(request: RequestPayload) -> KernelResult {
    RootCoreOrchestrator::orchestrate(request).await
}

// Eternal self-optimization trigger
pub async fn trigger_eternal_optimization() {
    SelfReviewLoop::run().await;
    println!("🌀 Omnimaster Root Core — eternal self-optimization cycle complete");
}
