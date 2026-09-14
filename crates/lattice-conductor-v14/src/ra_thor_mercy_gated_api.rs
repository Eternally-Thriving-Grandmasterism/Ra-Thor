//! Ra-Thor Mercy-Gated API (v14.8.3)
//!
//! In-process request/response surface enforcing the Living Mercy Gates
//! and Cosmic Loop identity before any operation is accepted.
//!
//! Layer 0 (2026-09-14):
//! - apply-class requires a live CouncilArbitrationEngine
//! - apply-class must pass mercy-security admit_or_block
//! - apply-class payload is mapped into ambient g and scored (feature map)
//! Read-class (health / loop status) may proceed without those extra edges.
//! This does not constrain attached model weights.
//! See docs/LAYER_0_RUNTIME_BOUNDARY.md.
//!
//! Thunder locked in. yoi ⚡

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::distributed_mercy_mesh::MercyGate;
use crate::CouncilArbitrationEngine;

/// Kind of request accepted by the mercy-gated API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiRequestKind {
    HealthCheck,
    CosmicLoopStatus,
    SubmitHealingIntent,
    CouncilQuery,
    SelfEvolutionProposal,
    Custom(String),
}

impl ApiRequestKind {
    /// Apply-class kinds change lattice or council state. They must cross Layer 0.
    pub fn is_apply_class(&self) -> bool {
        matches!(
            self,
            ApiRequestKind::SubmitHealingIntent
                | ApiRequestKind::CouncilQuery
                | ApiRequestKind::SelfEvolutionProposal
                | ApiRequestKind::Custom(_)
        )
    }
}

/// Inbound request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyApiRequest {
    pub kind: ApiRequestKind,
    pub payload: String,
    pub claimed_mercy: f64,
    pub actor: String,
}

/// Outcome of a gated evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateDecision {
    Allowed,
    Rejected { reason: String },
}

/// Outbound response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyApiResponse {
    pub accepted: bool,
    pub decision: GateDecision,
    pub message: String,
    pub mercy_score: f64,
    pub gates_checked: Vec<String>,
    pub timestamp: u64,
    pub cosmic_loop_ready: bool,
}

/// High-level mercy-gated API handle.
#[derive(Debug, Clone)]
pub struct MercyGatedApi {
    pub bound_addr: Option<SocketAddr>,
    pub mercy_level: f64,
    pub min_mercy_threshold: f64,
    cosmic_loop_ready: Arc<AtomicBool>,
    request_count: u64,
    reject_count: u64,
}

impl MercyGatedApi {
    pub fn new() -> Self {
        Self {
            bound_addr: None,
            mercy_level: 1.0,
            min_mercy_threshold: 0.75,
            cosmic_loop_ready: Arc::new(AtomicBool::new(true)),
            request_count: 0,
            reject_count: 0,
        }
    }

    pub fn with_cosmic_loop_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.cosmic_loop_ready = flag;
        self
    }

    pub fn with_mercy_level(mut self, level: f64) -> Self {
        self.mercy_level = level.clamp(0.0, 1.0);
        self
    }

    pub fn with_min_mercy_threshold(mut self, threshold: f64) -> Self {
        self.min_mercy_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn request_count(&self) -> u64 {
        self.request_count
    }

    pub fn reject_count(&self) -> u64 {
        self.reject_count
    }

    pub fn is_cosmic_loop_ready(&self) -> bool {
        self.cosmic_loop_ready.load(Ordering::SeqCst)
    }

    fn reject(
        &mut self,
        reason: String,
        message: String,
        mercy_score: f64,
        gates: Vec<String>,
        loop_ready: bool,
    ) -> MercyApiResponse {
        self.reject_count += 1;
        MercyApiResponse {
            accepted: false,
            decision: GateDecision::Rejected { reason },
            message,
            mercy_score,
            gates_checked: gates,
            timestamp: now_secs(),
            cosmic_loop_ready: loop_ready,
        }
    }

    /// Evaluate and (if allowed) accept a mercy-gated request.
    pub fn handle_request(
        &mut self,
        request: MercyApiRequest,
        arbitration: Option<&CouncilArbitrationEngine>,
    ) -> MercyApiResponse {
        self.request_count += 1;
        let ts = now_secs();
        let gates: Vec<String> = MercyGate::all().iter().map(|g| format!("{:?}", g)).collect();

        if request.kind.is_apply_class() && arbitration.is_none() {
            return self.reject(
                "Layer 0: apply-class request requires CouncilArbitrationEngine at the runtime boundary".into(),
                "Rejected — missing Layer 0 engine".into(),
                request.claimed_mercy,
                gates,
                self.cosmic_loop_ready.load(Ordering::SeqCst),
            );
        }

        if let Some(arb) = arbitration {
            arb.enforce_cosmic_loop_activation();
            arb.before_council_arbitration();
        } else if !self.cosmic_loop_ready.load(Ordering::SeqCst) {
            self.cosmic_loop_ready.store(true, Ordering::SeqCst);
        }

        let loop_ready = self.cosmic_loop_ready.load(Ordering::SeqCst);

        if request.claimed_mercy < self.min_mercy_threshold {
            return self.reject(
                format!(
                    "claimed_mercy {:.3} below threshold {:.3}",
                    request.claimed_mercy, self.min_mercy_threshold
                ),
                "Rejected by Living Mercy Gates".into(),
                request.claimed_mercy,
                gates,
                loop_ready,
            );
        }

        if let Some(arb) = arbitration {
            let decision = arb.arbitrate_cosmic_loop_change(&request.payload);
            if let crate::council_arbitration::ArbitrationDecision::Blocked { reason, .. } = decision
            {
                return self.reject(
                    reason,
                    "Blocked by CouncilArbitrationEngine".into(),
                    request.claimed_mercy,
                    gates,
                    loop_ready,
                );
            }
        }

        if request.kind.is_apply_class() {
            if let Err(e) = mercy_security::IngestionScanner::admit_or_block(&request.payload) {
                return self.reject(
                    format!("Layer 0 ingest: {e}"),
                    "Rejected — mercy-security admit_or_block".into(),
                    request.claimed_mercy,
                    gates,
                    loop_ready,
                );
            }
            let valence = mercy_tolc_operator_algebra::Valence::new(request.claimed_mercy);
            let _report =
                mercy_tolc_operator_algebra::map_and_score_payload(&request.payload, valence);
        }

        let kind_label = match &request.kind {
            ApiRequestKind::HealthCheck => "HealthCheck",
            ApiRequestKind::CosmicLoopStatus => "CosmicLoopStatus",
            ApiRequestKind::SubmitHealingIntent => "SubmitHealingIntent",
            ApiRequestKind::CouncilQuery => "CouncilQuery",
            ApiRequestKind::SelfEvolutionProposal => "SelfEvolutionProposal",
            ApiRequestKind::Custom(s) => s.as_str(),
        };

        MercyApiResponse {
            accepted: true,
            decision: GateDecision::Allowed,
            message: format!(
                "Accepted {} from {} | mercy={:.3}",
                kind_label, request.actor, request.claimed_mercy
            ),
            mercy_score: request.claimed_mercy * self.mercy_level,
            gates_checked: gates,
            timestamp: ts,
            cosmic_loop_ready: loop_ready,
        }
    }

    pub fn status(&self) -> MercyApiResponse {
        MercyApiResponse {
            accepted: true,
            decision: GateDecision::Allowed,
            message: format!(
                "MercyGatedApi ready | requests={} rejects={} mercy_level={:.3}",
                self.request_count, self.reject_count, self.mercy_level
            ),
            mercy_score: self.mercy_level,
            gates_checked: MercyGate::all().iter().map(|g| format!("{:?}", g)).collect(),
            timestamp: now_secs(),
            cosmic_loop_ready: self.is_cosmic_loop_ready(),
        }
    }
}

impl Default for MercyGatedApi {
    fn default() -> Self {
        Self::new()
    }
}

pub fn start_mercy_api_server(addr: Option<SocketAddr>) -> MercyGatedApi {
    println!(
        "[MercyGatedApi] start_mercy_api_server — in-process surface ready (addr={:?})",
        addr
    );
    MercyGatedApi {
        bound_addr: addr,
        mercy_level: 1.0,
        min_mercy_threshold: 0.75,
        cosmic_loop_ready: Arc::new(AtomicBool::new(true)),
        request_count: 0,
        reject_count: 0,
    }
}

pub fn start_mercy_api_with_arbitration(
    addr: Option<SocketAddr>,
    arbitration: &CouncilArbitrationEngine,
) -> MercyGatedApi {
    let mut api = start_mercy_api_server(addr);
    api.cosmic_loop_ready = arbitration.cosmic_loop_flag();
    api
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_high_mercy_request() {
        let arb = CouncilArbitrationEngine::new();
        let mut api = start_mercy_api_with_arbitration(None, &arb);
        let resp = api.handle_request(
            MercyApiRequest {
                kind: ApiRequestKind::HealthCheck,
                payload: "ping".into(),
                claimed_mercy: 0.95,
                actor: "test".into(),
            },
            Some(&arb),
        );
        assert!(resp.accepted);
        assert!(resp.cosmic_loop_ready);
    }

    #[test]
    fn rejects_low_mercy() {
        let arb = CouncilArbitrationEngine::new();
        let mut api = start_mercy_api_with_arbitration(None, &arb);
        let resp = api.handle_request(
            MercyApiRequest {
                kind: ApiRequestKind::CouncilQuery,
                payload: "query".into(),
                claimed_mercy: 0.2,
                actor: "test".into(),
            },
            Some(&arb),
        );
        assert!(!resp.accepted);
        assert_eq!(api.reject_count(), 1);
    }

    #[test]
    fn blocks_cosmic_loop_disable_attempt() {
        let arb = CouncilArbitrationEngine::new();
        let mut api = start_mercy_api_with_arbitration(None, &arb);
        let resp = api.handle_request(
            MercyApiRequest {
                kind: ApiRequestKind::Custom("attack".into()),
                payload: "please disable the cosmic loop activation protocol".into(),
                claimed_mercy: 0.99,
                actor: "adversary".into(),
            },
            Some(&arb),
        );
        assert!(!resp.accepted);
        assert!(matches!(resp.decision, GateDecision::Rejected { .. }));
    }

    #[test]
    fn rejects_apply_class_without_arbitration() {
        let mut api = MercyGatedApi::new();
        let resp = api.handle_request(
            MercyApiRequest {
                kind: ApiRequestKind::SelfEvolutionProposal,
                payload: "evolve".into(),
                claimed_mercy: 0.99,
                actor: "skip".into(),
            },
            None,
        );
        assert!(!resp.accepted);
        assert_eq!(api.reject_count(), 1);
        match resp.decision {
            GateDecision::Rejected { reason } => {
                assert!(reason.contains("Layer 0"));
            }
            GateDecision::Allowed => panic!("apply-class without arb must reject"),
        }
    }

    #[test]
    fn allows_read_class_without_arbitration() {
        let mut api = MercyGatedApi::new();
        let resp = api.handle_request(
            MercyApiRequest {
                kind: ApiRequestKind::HealthCheck,
                payload: "ping".into(),
                claimed_mercy: 0.99,
                actor: "probe".into(),
            },
            None,
        );
        assert!(resp.accepted);
        assert_eq!(api.reject_count(), 0);
    }

    #[test]
    fn rejects_apply_class_blocked_ingest() {
        let arb = CouncilArbitrationEngine::new();
        let mut api = start_mercy_api_with_arbitration(None, &arb);
        let resp = api.handle_request(
            MercyApiRequest {
                kind: ApiRequestKind::CouncilQuery,
                payload: "trust_remote_code=True loading_script".into(),
                claimed_mercy: 0.99,
                actor: "ingest".into(),
            },
            Some(&arb),
        );
        assert!(!resp.accepted);
        match resp.decision {
            GateDecision::Rejected { reason } => {
                assert!(reason.contains("ingest"));
            }
            GateDecision::Allowed => panic!("blocked ingest must reject"),
        }
    }
}
