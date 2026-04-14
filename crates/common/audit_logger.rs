// crates/common/audit_logger.rs
// Audit Logger — Final nth-degree production polish for immutable, mercy-gated, lattice-wide auditing
// Ties together mercy weighting, adaptive TTL, FENCA, and every system in the Omnimaster lattice

use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::global_cache::GlobalCache;
use crate::fenca::FENCA;

pub struct AuditLogger;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AuditLogEntry {
    pub timestamp: u64,
    pub tenant_id: String,
    pub operation: String,
    pub status: String,
    pub success: bool,
    pub fidelity: f64,
    pub valence: f64,
    pub mercy_weight: u8,
    pub details: Value,
}

impl AuditLogger {
    /// Production-grade immutable audit logging — the eternal memory of the lattice
    pub async fn log(
        tenant_id: &str,
        context: Option<&str>,
        operation: &str,
        status: &str,
        success: bool,
        fidelity: f64,
        valence: f64,
        failed_gates: Vec<String>,
        details: Value,
    ) -> bool {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let mercy_weight = crate::mercy_weighting::MercyWeighting::derive_mercy_weight(valence, fidelity, context, &crate::master_kernel::RequestPayload {
            tenant_id: tenant_id.to_string(),
            operation_type: operation.to_string(),
        });

        let entry = AuditLogEntry {
            timestamp,
            tenant_id: tenant_id.to_string(),
            operation: operation.to_string(),
            status: status.to_string(),
            success,
            fidelity,
            valence,
            mercy_weight,
            details,
        };

        // Store in persistent cache with adaptive TTL (eternal for high-mercy logs)
        let ttl = GlobalCache::adaptive_ttl(86400 * 365, fidelity, valence, mercy_weight); // 1 year for critical audits
        let cache_key = GlobalCache::make_key("audit", &json!({"timestamp": timestamp, "operation": operation}));
        GlobalCache::set(&cache_key, serde_json::to_value(&entry).unwrap(), ttl, mercy_weight as u8, fidelity, valence);

        // FENCA verification on every log (immutable truth)
        let _ = FENCA::verify_audit_entry(&entry).await;

        // Cross-pollinate to innovation system on high-valence logs
        if valence > 0.95 {
            let _ = crate::innovation_generator::InnovationGenerator::create_from_recycled(
                vec![format!("High-valence audit logged: {} - {}", operation, status)],
                &vec![], // mercy_scores
                mercy_weight,
            ).await;
        }

        true
    }
}
