// core/root_core_orchestrator.rs
// Root Core — The Leader Agent of Ra-Thor
// Orchestrates all Sub-Cores (crates), reviews /docs folder, recycles ideas, generates innovations, and self-improves
// Everything remains FENCA-first, mercy-gated, and sovereign

use crate::master_kernel::{RequestPayload, KernelResult};
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use crate::mercy_weighting::MercyWeighting;
use crate::audit_logger::AuditLogger;
use crate::global_cache::GlobalCache;
use serde_json::Value;

pub struct RootCoreOrchestrator;

impl RootCoreOrchestrator {
    /// Main Leader Agent entry point — orchestrates everything
    pub async fn orchestrate(request: RequestPayload) -> KernelResult {
        let start = std::time::Instant::now();

        // 1. FENCA — primordial truth gate (always first)
        let fenca_result = FENCA::verify_tenant_scoped(&request, &request.tenant_id);
        if !fenca_result.is_verified() {
            let _ = AuditLogger::log(&request.tenant_id, None, "fenca_failed", &request.operation_type, false, 0.0, 0.0, vec!["fenca".to_string()], Value::Null).await;
            return fenca_result.gentle_reroute();
        }

        // 2. Mercy Engine — ethical core
        let mercy_scores = MercyEngine::evaluate_deep_with_tenant(&request, &request.tenant_id);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);
        if !mercy_scores.all_gates_pass() {
            let _ = AuditLogger::log(&request.tenant_id, None, "mercy_failed", &request.operation_type, false, fenca_result.fidelity(), valence, mercy_scores.failed_gates(), Value::Null).await;
            return MercyEngine::gentle_reroute_with_preservation(&request, &mercy_scores);
        }

        // 3. Mercy Weight (for delegation tuning)
        let mercy_weight = MercyWeighting::derive_mercy_weight(valence, fenca_result.fidelity(), None, &request);

        // 4. Delegate to appropriate Sub-Core
        let result = match request.operation_type.as_str() {
            "access" | "rebac" | "rbac" | "abac" => crates::access::handle(request.clone(), mercy_weight).await,
            "quantum" | "vqc" | "entanglement" | "distillation" | "teleportation" => crates::quantum::handle(request.clone(), mercy_weight).await,
            "mercy" | "valence" | "gate_scoring" => crates::mercy::handle(request.clone(), mercy_weight).await,
            "persistence" | "quota" | "indexeddb" => crates::persistence::handle(request.clone(), mercy_weight).await,
            "orchestration" | "multi_user" => crates::orchestration::handle(request.clone(), mercy_weight).await,
            "cache" | "ttl" => crates::cache::handle(request.clone(), mercy_weight).await,
            _ => crates::common::default_handle(request.clone(), mercy_weight).await,
        };

        // 5. Self-review & Innovation Loop (runs periodically or on high-valence operations)
        if valence > 0.95 || rand::random::<f64>() < 0.1 {
            Self::self_review_and_innovate().await;
        }

        // 6. Immutable audit
        let _ = AuditLogger::log(
            &request.tenant_id,
            None,
            "root_core_orchestration",
            &result.status,
            true,
            fenca_result.fidelity(),
            valence,
            vec![],
            serde_json::json!({
                "duration_ms": start.elapsed().as_millis(),
                "mercy_weight": mercy_weight
            }),
        ).await;

        result
    }

    /// Leader Agent self-review: scans /docs, recycles ideas, generates innovations
    pub async fn self_review_and_innovate() {
        let codices = CodexLoader::scan_docs_folder().await;

        for codex in codices {
            if FENCA::verify_codex(&codex).await.is_verified() {
                let mercy_scores = MercyEngine::evaluate_codex(&codex);
                if mercy_scores.all_gates_pass() {
                    let recycled = IdeaRecycler::extract_ideas(&codex);
                    let innovation = InnovationGenerator::create_from_recycled(recycled, &mercy_scores).await;
                    if let Some(inn) = innovation {
                        Self::delegate_innovation(inn).await;
                    }
                }
            }
        }

        // Self-optimization
        MercyWeighting::tune_from_recent_audit_logs().await;
    }

    /// Delegate innovation to the appropriate Sub-Core
    pub async fn delegate_innovation(innovation: Innovation) {
        match innovation.target {
            Target::Quantum => crates::quantum::apply_innovation(innovation).await,
            Target::Mercy => crates::mercy::apply_innovation(innovation).await,
            Target::Access => crates::access::apply_innovation(innovation).await,
            Target::Persistence => crates::persistence::apply_innovation(innovation).await,
            _ => crates::kernel::apply_innovation(innovation).await,
        }
    }
}
