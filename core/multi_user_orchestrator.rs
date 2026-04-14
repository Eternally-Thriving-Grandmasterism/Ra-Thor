// core/multi_user_orchestrator.rs
// Central Multi-User Orchestrator — the single elegant entry point for all enterprise requests
// Fully revised and polished: tenant-isolated, mercy-gated, FENCA-first, RBAC/ReBAC/ABAC hybrid, quota enforcement, audit logging, and Master Sovereign Kernel routing

use crate::master_kernel::{ra_thor_sovereign_master_kernel, RequestPayload, KernelResult};
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use crate::rbac::RBAC;
use crate::rebac::RelationshipGraph;
use crate::hybrid_access::HybridAccess;
use crate::resource_quota::ResourceQuotaEnforcer;
use crate::audit_logger::AuditLogger;
use crate::global_cache::GlobalCache;
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

        // 1. Tenant Isolation & Session Validation
        let tenant_id = &session.tenant_id;

        // 2. Hybrid RBAC + ReBAC + ABAC check (cached, tenant-scoped)
        if let Err(reroute) = HybridAccess::check(&session, &request) {
            let _ = AuditLogger::log(tenant_id, Some(&session.user_id), "access_denied", &request.operation_type, false, 0.0, 0.0, vec!["hybrid_access".to_string()], Value::Null).await;
            return reroute;
        }

        // 3. Resource Quota Enforcement (mercy-aware)
        if let Err(reroute) = ResourceQuotaEnforcer::enforce(tenant_id, &request) {
            let _ = AuditLogger::log(tenant_id, Some(&session.user_id), "quota_exceeded", &request.operation_type, false, 0.0, 0.0, vec!["resource_quota".to_string()], Value::Null).await;
            return reroute;
        }

        // 4. FENCA — primordial truth gate (always first after access checks)
        let fenca_result = FENCA::verify_tenant_scoped(&request, tenant_id);
        if !fenca_result.is_verified() {
            let _ = AuditLogger::log(tenant_id, Some(&session.user_id), "fenca_failed", &request.operation_type, false, 0.0, 0.0, vec!["fenca".to_string()], Value::Null).await;
            return fenca_result.gentle_reroute();
        }

        // 5. Mercy Engine — ethical core
        let mercy_scores = MercyEngine::evaluate_deep_with_tenant(&request, tenant_id);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);
        if !mercy_scores.all_gates_pass() {
            let _ = AuditLogger::log(tenant_id, Some(&session.user_id), "mercy_failed", &request.operation_type, false, fenca_result.fidelity(), valence, mercy_scores.failed_gates(), Value::Null).await;
            return MercyEngine::gentle_reroute_with_preservation(&request, &mercy_scores);
        }

        // 6. Route to Master Sovereign Kernel (the heart)
        let kernel_result = ra_thor_sovereign_master_kernel(request, 1_000_000, 2);

        // 7. Immutable Audit Log (always recorded)
        let _ = AuditLogger::log(
            tenant_id,
            Some(&session.user_id),
            "kernel_execution",
            &kernel_result.status,
            true,
            fenca_result.fidelity(),
            valence,
            vec![],
            serde_json::json!({
                "duration_ms": start_time.elapsed().as_millis(),
                "ghz_fidelity": kernel_result.ghz_fidelity,
                "valence": kernel_result.valence
            }),
        ).await;

        kernel_result
    }
}
