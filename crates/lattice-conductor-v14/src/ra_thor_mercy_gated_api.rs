//! Ra-Thor Mercy-Gated API (v14.8.3)
//!
//! In-process request/response surface enforcing the 7 Living Mercy Gates
//! and Cosmic Loop identity before any operation is accepted.
//!
//! Optional Axum HTTP binding is available behind the `web-demo` feature.
//! Default build stays dependency-light (no network stack required).
//!
//! Thunder locked in. yoi ⚡

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::distributed_mercy_mesh::MercyGate;
use crate::CouncilArbitrationEngine;

/// Kind of request accepted by the mercy-gated API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiRequestKind {
    HealthCheck,
    CosmicLoopStatus,
    SubmitHealingIntent,
    CouncilQuery,
    SelfEvolutionProposal,
    Custom(String),
}

/// Inbound request.
#[derive(Debug, Clone)]
pub struct MercyApiRequest {
    pub kind: ApiRequestKind,
    pub payload: String,
    pub claimed_mercy: f64,
    pub actor: String,
}

/// Outcome of a gated evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateDecision {
    Allowed,
    Rejected { reason: String },
}

/// Outbound response.
#[derive(Debug, Clone)]
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
    /// Shared Cosmic Loop flag (wired from CouncilArbitrationEngine when available).
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

    /// Wire the shared Cosmic Loop flag from the arbitration engine.
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

    /// Evaluate and (if allowed) accept a mercy-gated request.
    pub fn handle_request(
        &mut self,
        request: MercyApiRequest,
        arbitration: Option<&CouncilArbitrationEngine>,
    ) -> MercyApiResponse {
        self.request_count += 1;
        let ts = now_secs();
        let gates: Vec<String> = MercyGate::all().iter().map(|g| format!("{:?}", g)).collect();

        // 1. Cosmic Loop is mandatory identity
        if let Some(arb) = arbitration {
            arb.enforce_cosmic_loop_activation();
            arb.before_council_arbitration();
        } else if !self.cosmic_loop_ready.load(Ordering::SeqCst) {
            self.cosmic_loop_ready.store(true, Ordering::SeqCst);
        }

        let loop_ready = self.cosmic_loop_ready.load(Ordering::SeqCst);

        // 2. Mercy threshold gate
        if request.claimed_mercy < self.min_mercy_threshold {
            self.reject_count += 1;
            return MercyApiResponse {
                accepted: false,
                decision: GateDecision::Rejected {
                    reason: format!(
                        "claimed_mercy {:.3} below threshold {:.3}",
                        request.claimed_mercy, self.min_mercy_threshold
                    ),
                },
                message: "Rejected by Living Mercy Gates".into(),
                mercy_score: request.claimed_mercy,
                gates_checked: gates,
                timestamp: ts,
                cosmic_loop_ready: loop_ready,
            };
        }

        // 3. Keyword hardening against Cosmic Loop weakening
        if let Some(arb) = arbitration {
            let decision = arb.arbitrate_cosmic_loop_change(&request.payload);
            if let crate::council_arbitration::ArbitrationDecision::Blocked { reason, .. } = decision
            {
                self.reject_count += 1;
                return MercyApiResponse {
                    accepted: false,
                    decision: GateDecision::Rejected { reason },
                    message: "Blocked by CouncilArbitrationEngine".into(),
                    mercy_score: request.claimed_mercy,
                    gates_checked: gates,
                    timestamp: ts,
                    cosmic_loop_ready: loop_ready,
                };
            }
        }

        // 4. Accept
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

    /// Convenience: status snapshot.
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

/// Start a mercy-gated API handle.
///
/// Without the `web-demo` feature this is an in-process handle only
/// (no socket bind). With `web-demo`, an Axum listener can be layered later.
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

/// Start API wired to an existing arbitration engine (shared Cosmic Loop flag).
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
}
