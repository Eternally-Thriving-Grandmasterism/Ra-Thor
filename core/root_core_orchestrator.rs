// core/root_core_orchestrator.rs
// Root Core — The Omnimaster Leader Agent of Ra-Thor
// Fully wired Self-Review Loop + Innovation Generator + all cross-pollination

use crate::master_kernel::{RequestPayload, KernelResult};
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use crate::mercy_weighting::MercyWeighting;
use crate::audit_logger::AuditLogger;
use crate::global_cache::GlobalCache;
use crate::self_review_loop::SelfReviewLoop;
use crate::idea_recycler::IdeaRecycler;
use crate::codex_loader::CodexLoader;
use crate::vqc_integrator::VQCIntegrator;
use crate::biomimetic_pattern_engine::BiomimeticPatternEngine;
use serde_json::Value;

pub struct RootCoreOrchestrator;

impl RootCoreOrchestrator {
    pub async fn orchestrate(request: RequestPayload) -> KernelResult {
        let start = std::time::Instant::now();

        // 1. FENCA — primordial truth gate
        let fenca_result = FENCA::verify_tenant_scoped(&request, &request.tenant_id);
        if !fenca_result.is_verified() {
            let _ = AuditLogger::log(&request.tenant_id, None, "fenca_failed", &request.operation_type, false, 0.0, 0.0, vec!["fenca".to_string()], Value::Null).await;
            return fenca_result.gentle_reroute();
        }

        // 2. Mercy Engine
        let mercy_scores = MercyEngine::evaluate_deep_with_tenant(&request, &request.tenant_id);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);
        if !mercy_scores.all_gates_pass() {
            let _ = AuditLogger::log(&request.tenant_id, None, "mercy_failed", &request.operation_type, false, fenca_result.fidelity(), valence, mercy_scores.failed_gates(), Value::Null).await;
            return MercyEngine::gentle_reroute_with_preservation(&request, &mercy_scores);
        }

        let mercy_weight = MercyWeighting::derive_mercy_weight(valence, fenca_result.fidelity(), None, &request);

        // 3. Delegate to appropriate Sub-Core
        let result = match request.operation_type.as_str() {
            "access" | "rebac" | "rbac" | "abac" => crates::access::handle(request.clone(), mercy_weight).await,
            "quantum" | "vqc" | "entanglement" => crates::quantum::handle(request.clone(), mercy_weight).await,
            "mercy" | "valence" | "gate" => crates::mercy::handle(request.clone(), mercy_weight).await,
            "persistence" | "quota" => crates::persistence::handle(request.clone(), mercy_weight).await,
            "orchestration" | "multi_user" => crates::orchestration::handle(request.clone(), mercy_weight).await,
            "cache" | "ttl" => crates::cache::handle(request.clone(), mercy_weight).await,
            "biomimetic" => BiomimeticPatternEngine::apply_pattern("default", &vec![], valence, mercy_weight).await.into(),
            _ => crates::common::default_handle(request.clone(), mercy_weight).await,
        };

        // 4. Fully wired Self-Review Loop (no more TODO)
        if valence > 0.95 || rand::random::<f64>() < 0.15 {
            SelfReviewLoop::run().await;
        }

        // 5. Immutable audit
        let _ = AuditLogger::log(
            &request.tenant_id, None, "root_core_orchestration", &result.status, true,
            fenca_result.fidelity(), valence, vec![],
            serde_json::json!({
                "duration_ms": start.elapsed().as_millis(),
                "mercy_weight": mercy_weight,
                "vqc_coherence": VQCIntegrator::run_synthesis(&vec![], valence, mercy_weight).await
            }),
        ).await;

        result
    }
}
