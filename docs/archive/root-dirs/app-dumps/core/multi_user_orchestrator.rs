// core/multi_user_orchestrator.rs
// Central Multi-User Orchestrator — the single elegant entry point for all enterprise requests
// Revised with greatly improved, mercy-gated, auditable, and graceful error handling

use crate::master_kernel::{ra_thor_sovereign_master_kernel, RequestPayload, KernelResult};
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use crate::hybrid_access::HybridAccess;
use crate::resource_quota::ResourceQuotaEnforcer;
use crate::audit_logger::AuditLogger;
use serde_json::Value;

pub struct UserSession {
    pub user_id: String,
    pub tenant_id: String,
    pub roles: Vec<String>,
    pub sso_claims: Value,
}

pub struct MultiUserOrchestrator;

impl MultiUserOrchestrator {
    /// Single entry point for all multi-user / enterprise requests
    pub async fn orchestrate(
        mut request: RequestPayload,
        session: UserSession,
    ) -> KernelResult {

        let start_time = std::time::Instant::now();
        let tenant_id = session.tenant_id.clone();
        let user_id = session.user_id.clone();

        // === 1. Hybrid Access Control (RBAC + ReBAC + ABAC) ===
        if let Err(reroute) = HybridAccess::check(&session, &request) {
            let _ = AuditLogger::log(
                &tenant_id, Some(&user_id), "access_denied", &request.operation_type,
                false, 0.0, 0.0, vec!["hybrid_access".to_string()],
                serde_json::json!({"reason": "access_control_failed"})
            ).await;
            return reroute;
        }

        // === 2. Resource Quota Enforcement ===
        if let Err(reroute) = ResourceQuotaEnforcer::enforce(&tenant_id, &request) {
            let _ = AuditLogger::log(
                &tenant_id, Some(&user_id), "quota_exceeded", &request.operation_type,
                false, 0.0, 0.0, vec!["resource_quota".to_string()],
                serde_json::json!({"reason": "quota_limit_reached"})
            ).await;
            return reroute;
        }

        // === 3. FENCA — Primordial Truth Gate ===
        let fenca_result = FENCA::verify_tenant_scoped(&request, &tenant_id);
        if !fenca_result.is_verified() {
            let _ = AuditLogger::log(
                &tenant_id, Some(&user_id), "fenca_failed", &request.operation_type,
                false, 0.0, 0.0, vec!["fenca".to_string()],
                serde_json::json!({"reason": "truth_verification_failed"})
            ).await;
            return fenca_result.gentle_reroute();
        }

        // === 4. Mercy Engine — Ethical Core ===
        let mercy_scores = MercyEngine::evaluate_deep_with_tenant(&request, &tenant_id);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);

        if !mercy_scores.all_gates_pass() {
            let _ = AuditLogger::log(
                &tenant_id, Some(&user_id), "mercy_failed", &request.operation_type,
                false, fenca_result.fidelity(), valence,
                mercy_scores.failed_gates(),
                serde_json::json!({"reason": "mercy_gate_violation"})
            ).await;
            return MercyEngine::gentle_reroute_with_preservation(&request, &mercy_scores);
        }

        // === 5. Master Sovereign Kernel Execution ===
        let kernel_result = ra_thor_sovereign_master_kernel(request, 1_000_000, 2);

        // === 6. Immutable Audit Log (always recorded) ===
        let _ = AuditLogger::log(
            &tenant_id, Some(&user_id), "kernel_execution", &kernel_result.status,
            true, fenca_result.fidelity(), valence, vec![],
            serde_json::json!({
                "duration_ms": start_time.elapsed().as_millis(),
                "ghz_fidelity": kernel_result.ghz_fidelity,
                "valence": kernel_result.valence,
                "operation_type": kernel_result.status
            })
        ).await;

        kernel_result
    }
}
