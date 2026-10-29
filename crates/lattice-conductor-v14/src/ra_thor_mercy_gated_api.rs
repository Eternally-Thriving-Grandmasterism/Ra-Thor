//! Ra-Thor Mercy-Gated API (v14.8.3)
//!
//! In-process request/response surface enforcing the Living Mercy Gates
//! and Cosmic Loop identity before any operation is accepted.
//!
//! Layer 0 (2026-09-14):
//! - apply-class requires a live CouncilArbitrationEngine
//! - apply-class must pass mercy-security admit_or_block
//! - apply-class payload is mapped into ambient g and scored (feature map)
//! - apply-class then calls `bounded_evolution_step` (E4), fail-closed
//! Read-class (health / loop status) may proceed without those extra edges.
//! A `handle_request` return that never calls E4 records `wrap not counted`.
//! Engine and threshold still run before admit, so this is not E1→E2→E3→E4.
//! This does not constrain attached model weights.
//! See docs/LAYER_0_RUNTIME_BOUNDARY.md and docs/WRAP_FOUR_EDGES.md.
//!
//! Thunder locked in. yoi ⚡

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::distributed_mercy_mesh::MercyGate;
use crate::evidence_chain::{
    payload_digest, EvidenceChain, EvidenceDraft, EvidenceError, EvidenceKind,
};
use crate::inspect_sae::{
    InspectGateResult, InspectMode, InspectPacket, InspectRecorder, SaeError,
};
use crate::lipschitz_gate::{LipschitzBall, LipschitzCheck, LipschitzError, LipschitzGate, Theta};
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

/// Phrase recorded when a `handle_request` path does not call E4.
pub const WRAP_NOT_COUNTED: &str = "wrap not counted";

/// E4 account for one `handle_request`.
///
/// `counted` stays false on this tip. Engine and threshold still run before
/// admit, so E1 then E2 then E3 then E4 is not claimed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WrapAccount {
    pub e4_called: bool,
    /// True only when E1, E2, E3, and E4 have each fired in that order.
    pub counted: bool,
    /// `Some(WRAP_NOT_COUNTED)` when E4 did not run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub miss: Option<String>,
}

impl WrapAccount {
    fn miss() -> Self {
        Self {
            e4_called: false,
            counted: false,
            miss: Some(WRAP_NOT_COUNTED.into()),
        }
    }

    /// E4 ran. The ordered four-edge wrap is still not claimed.
    fn e4_not_counted() -> Self {
        Self {
            e4_called: true,
            counted: false,
            miss: None,
        }
    }
}

impl Default for WrapAccount {
    fn default() -> Self {
        Self {
            e4_called: false,
            counted: false,
            miss: None,
        }
    }
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
    /// Hash of the evidential-face row for this apply-class decision, if emitted.
    #[serde(default)]
    pub evidence_hash: Option<String>,
    /// Hash of the inspect packet recorded on wrap, if any.
    #[serde(default)]
    pub inspect_packet_hash: Option<String>,
    /// Four-edge account. A missing E4 call is a miss, not a counted wrap.
    #[serde(default)]
    pub wrap_account: WrapAccount,
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
    evidence: Arc<Mutex<EvidenceChain>>,
    lipschitz: Arc<Mutex<LipschitzGate>>,
    inspect: Arc<Mutex<InspectRecorder>>,
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
            evidence: Arc::new(Mutex::new(EvidenceChain::new())),
            lipschitz: Arc::new(Mutex::new(LipschitzGate::new())),
            inspect: Arc::new(Mutex::new(InspectRecorder::default())),
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
            evidence_hash: None,
            inspect_packet_hash: None,
            wrap_account: WrapAccount::miss(),
        }
    }

    pub fn evidence_chain(&self) -> Arc<Mutex<EvidenceChain>> {
        Arc::clone(&self.evidence)
    }

    pub fn lipschitz_gate(&self) -> Arc<Mutex<LipschitzGate>> {
        Arc::clone(&self.lipschitz)
    }

    pub fn lipschitz_ball_installed(&self) -> bool {
        self.lipschitz
            .lock()
            .ok()
            .map(|g| g.current_ball().is_some())
            .unwrap_or(false)
    }

    pub fn install_lipschitz_ball(&self, ball: LipschitzBall) -> Result<(), LipschitzError> {
        let mut gate = self.lipschitz.lock().map_err(|_| LipschitzError::LockPoisoned)?;
        gate.install(ball)
    }

    pub fn freeze_lipschitz_ball(
        &self,
        theta0: Theta,
        margin_m: f64,
        lipschitz_l: f64,
    ) -> Result<(), LipschitzError> {
        let mut gate = self.lipschitz.lock().map_err(|_| LipschitzError::LockPoisoned)?;
        gate.freeze_new_ball(theta0, margin_m, lipschitz_l)
    }

    pub fn check_lipschitz(
        &self,
        theta: &Theta,
        actor: &str,
        timestamp: u64,
    ) -> Result<LipschitzCheck, LipschitzError> {
        let mut gate = self.lipschitz.lock().map_err(|_| LipschitzError::LockPoisoned)?;
        if gate.current_ball().is_none() {
            return Err(LipschitzError::MissingTheta0);
        }
        let check = gate.verify(theta);
        drop(gate);
        let directive = if check.accepted { "accept" } else { "reject" };
        let mut chain = self
            .evidence
            .lock()
            .map_err(|_| LipschitzError::LockPoisoned)?;
        chain.append(EvidenceDraft {
            subject: format!("lipschitz:{actor}"),
            input: payload_digest(&format!("{:?}", theta.0)),
            claim: check.reason.clone(),
            evidence_pointers: vec![
                format!("m={:?}", check.margin_m),
                format!("L={:?}", check.lipschitz_l),
                format!("r={:?}", check.radius),
                format!("displacement={:?}", check.displacement),
            ],
            directive: directive.into(),
            scope: "lattice-conductor-v14".into(),
            kind: EvidenceKind::Lipschitz,
            decision: directive.into(),
            timestamp,
            actor: actor.into(),
            auditor: "lipschitz-gate".into(),
        })
        .map_err(|_| LipschitzError::EvidenceMissing)?;
        Ok(check)
    }

    pub fn inspect_recorder(&self) -> Arc<Mutex<InspectRecorder>> {
        Arc::clone(&self.inspect)
    }

    pub fn set_inspect_mode(&self, mode: InspectMode) -> Result<(), SaeError> {
        let mut rec = self
            .inspect
            .lock()
            .map_err(|_| SaeError::LockPoisoned)?;
        rec.set_mode(mode);
        Ok(())
    }

    pub fn last_inspect_packet(&self) -> Option<InspectPacket> {
        self.inspect
            .lock()
            .ok()
            .and_then(|r| r.last().cloned())
    }

    pub fn record_wrap_inspect(
        &self,
        model_id: &str,
        text: &str,
    ) -> Result<InspectPacket, SaeError> {
        let mut rec = self
            .inspect
            .lock()
            .map_err(|_| SaeError::LockPoisoned)?;
        rec.record_text(
            model_id,
            crate::inspect_sae::WRAP_HOOK_SITE,
            text,
            Some(crate::inspect_sae::WRAP_CIRCUIT_ID.into()),
        )
    }

    pub fn finish_wrap_inspect(
        &self,
        gate: InspectGateResult,
        steering_applied: bool,
    ) -> Result<(), SaeError> {
        let mut rec = self
            .inspect
            .lock()
            .map_err(|_| SaeError::LockPoisoned)?;
        rec.update_last_gate(gate, steering_applied);
        Ok(())
    }

    /// Opportunity 3 attach: one Inspect row pointing at the packet + wrap hash.
    pub fn attach_inspect_evidence(
        &self,
        packet: &InspectPacket,
        wrap_hash: Option<&str>,
        actor: &str,
        timestamp: u64,
        accepted: bool,
    ) -> Result<String, EvidenceError> {
        let mut chain = self
            .evidence
            .lock()
            .map_err(|_| EvidenceError::LockPoisoned)?;
        let directive = if accepted { "accept" } else { "reject" };
        let rec = chain.append(EvidenceDraft {
            subject: format!("inspect:{actor}"),
            input: packet.packet_hash.clone(),
            claim: format!(
                "inspect {} gate={}",
                packet.hook_site,
                packet.gate_result.as_label()
            ),
            evidence_pointers: vec![
                packet.as_evidence_pointer(),
                format!("wrap:{}", wrap_hash.unwrap_or("none")),
                format!("backend:{}", packet.backend_id),
            ],
            directive: directive.into(),
            scope: "lattice-conductor-v14".into(),
            kind: EvidenceKind::Inspect,
            decision: directive.into(),
            timestamp,
            actor: actor.into(),
            auditor: "inspect-face".into(),
        })?;
        Ok(rec.hash)
    }

    fn emit_gate_record(
        &self,
        request: &MercyApiRequest,
        accepted: bool,
        reason: &str,
    ) -> Result<String, EvidenceError> {
        let mut chain = self
            .evidence
            .lock()
            .map_err(|_| EvidenceError::LockPoisoned)?;
        let kind = match &request.kind {
            ApiRequestKind::SelfEvolutionProposal => EvidenceKind::SelfEvolution,
            ApiRequestKind::Custom(_) => EvidenceKind::Wrap,
            ApiRequestKind::CouncilQuery => EvidenceKind::Council,
            _ => EvidenceKind::Tool,
        };
        let directive = if accepted { "accept" } else { "reject" };
        let rec = chain.append(EvidenceDraft {
            subject: format!("{}:{}", kind.as_str(), request.actor),
            input: payload_digest(&request.payload),
            claim: reason.to_string(),
            evidence_pointers: vec!["layer0".into(), "handle_request".into()],
            directive: directive.into(),
            scope: "lattice-conductor-v14".into(),
            kind,
            decision: directive.into(),
            timestamp: now_secs(),
            actor: request.actor.clone(),
            auditor: "evidence-face".into(),
        })?;
        Ok(rec.hash)
    }

    fn chain_gate(&self, hash: &str) -> Result<(), EvidenceError> {
        let chain = self
            .evidence
            .lock()
            .map_err(|_| EvidenceError::LockPoisoned)?;
        chain.gate_side_effect(Some(hash))
    }

    /// Evidential face: Allowed apply-class without a chained record is not apply.
    fn seal_apply(
        &mut self,
        request: &MercyApiRequest,
        mut resp: MercyApiResponse,
        reason: &str,
    ) -> MercyApiResponse {
        if !request.kind.is_apply_class() {
            return resp;
        }
        match self.emit_gate_record(request, resp.accepted, reason) {
            Ok(h) => {
                if resp.accepted && self.chain_gate(&h).is_err() {
                    let account = resp.wrap_account.clone();
                    let mut rejected = self.reject(
                        "missing or broken evidence chain".into(),
                        "Rejected — evidential face".into(),
                        request.claimed_mercy,
                        resp.gates_checked,
                        resp.cosmic_loop_ready,
                    );
                    rejected.wrap_account = account;
                    return rejected;
                }
                resp.evidence_hash = Some(h);
                resp
            }
            Err(e) => {
                if resp.accepted {
                    let account = resp.wrap_account.clone();
                    let mut rejected = self.reject(
                        format!("evidence: {e}"),
                        "Rejected — evidential face".into(),
                        request.claimed_mercy,
                        resp.gates_checked,
                        resp.cosmic_loop_ready,
                    );
                    rejected.wrap_account = account;
                    rejected
                } else {
                    resp
                }
            }
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
            let reason = "Layer 0: apply-class request requires CouncilArbitrationEngine at the runtime boundary";
            let resp = self.reject(
                reason.into(),
                "Rejected — missing Layer 0 engine".into(),
                request.claimed_mercy,
                gates,
                self.cosmic_loop_ready.load(Ordering::SeqCst),
            );
            return self.seal_apply(&request, resp, reason);
        }

        if let Some(arb) = arbitration {
            arb.enforce_cosmic_loop_activation();
            arb.before_council_arbitration();
        } else if !self.cosmic_loop_ready.load(Ordering::SeqCst) {
            self.cosmic_loop_ready.store(true, Ordering::SeqCst);
        }

        let loop_ready = self.cosmic_loop_ready.load(Ordering::SeqCst);

        if request.claimed_mercy < self.min_mercy_threshold {
            let reason = format!(
                "claimed_mercy {:.3} below threshold {:.3}",
                request.claimed_mercy, self.min_mercy_threshold
            );
            let resp = self.reject(
                reason.clone(),
                "Rejected by Living Mercy Gates".into(),
                request.claimed_mercy,
                gates,
                loop_ready,
            );
            return self.seal_apply(&request, resp, &reason);
        }

        if let Some(arb) = arbitration {
            let decision = arb.arbitrate_cosmic_loop_change(&request.payload);
            if let crate::council_arbitration::ArbitrationDecision::Blocked { reason, .. } = decision
            {
                let resp = self.reject(
                    reason.clone(),
                    "Blocked by CouncilArbitrationEngine".into(),
                    request.claimed_mercy,
                    gates,
                    loop_ready,
                );
                return self.seal_apply(&request, resp, &reason);
            }
        }

        if request.kind.is_apply_class() {
            if let Err(e) = mercy_security::IngestionScanner::admit_or_block(&request.payload) {
                let reason = format!("Layer 0 ingest: {e}");
                let resp = self.reject(
                    reason.clone(),
                    "Rejected — mercy-security admit_or_block".into(),
                    request.claimed_mercy,
                    gates,
                    loop_ready,
                );
                return self.seal_apply(&request, resp, &reason);
            }
            let valence = mercy_tolc_operator_algebra::Valence::new(request.claimed_mercy);
            let _report =
                mercy_tolc_operator_algebra::map_and_score_payload(&request.payload, valence);
            // E4 after admit, the existing threshold, and the projector.
            // `claimed_mercy` is passed through. `min_mercy_threshold` is not written.
            if !e4_bounded_step_allows(request.claimed_mercy) {
                let mut breaker =
                    sovereign_recovery::MercyGatedCircuitBreaker::new("apply-class");
                breaker.trip(request.claimed_mercy);
                let reason = "Layer 0 circuit: bounded_evolution_step rejected".to_string();
                let mut resp = self.reject(
                    reason.clone(),
                    "Rejected — sovereign-recovery bounded_evolution_step".into(),
                    request.claimed_mercy,
                    gates,
                    loop_ready,
                );
                resp.wrap_account = WrapAccount::e4_not_counted();
                return self.seal_apply(&request, resp, &reason);
            }
        }

        let wrap_account = if request.kind.is_apply_class() {
            WrapAccount::e4_not_counted()
        } else {
            WrapAccount::miss()
        };

        let kind_label = match &request.kind {
            ApiRequestKind::HealthCheck => "HealthCheck",
            ApiRequestKind::CosmicLoopStatus => "CosmicLoopStatus",
            ApiRequestKind::SubmitHealingIntent => "SubmitHealingIntent",
            ApiRequestKind::CouncilQuery => "CouncilQuery",
            ApiRequestKind::SelfEvolutionProposal => "SelfEvolutionProposal",
            ApiRequestKind::Custom(s) => s.as_str(),
        };

        let resp = MercyApiResponse {
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
            evidence_hash: None,
            inspect_packet_hash: None,
            wrap_account,
        };
        self.seal_apply(&request, resp, "accept")
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
            evidence_hash: None,
            inspect_packet_hash: None,
            wrap_account: WrapAccount::default(),
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
        evidence: Arc::new(Mutex::new(EvidenceChain::new())),
        lipschitz: Arc::new(Mutex::new(LipschitzGate::new())),
        inspect: Arc::new(Mutex::new(InspectRecorder::default())),
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

/// Call E4. Pending or false is fail-closed.
/// Does not read or write `min_mercy_threshold`.
fn e4_bounded_step_allows(mercy_alignment: f64) -> bool {
    let protocol = sovereign_recovery::SovereignRecoveryProtocol::new();
    let step = protocol.bounded_evolution_step("handle_request", mercy_alignment);
    match poll_now(step) {
        Some(ok) => ok,
        None => false,
    }
}

/// `bounded_evolution_step` does not await. A later `.await` that stays pending fails closed.
fn poll_now<T>(fut: impl std::future::Future<Output = T>) -> Option<T> {
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    unsafe fn clone(data: *const ()) -> RawWaker {
        RawWaker::new(data, &VTABLE)
    }
    unsafe fn wake(_: *const ()) {}
    unsafe fn wake_by_ref(_: *const ()) {}
    unsafe fn drop(_: *const ()) {}

    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    let mut fut = std::pin::pin!(fut);
    match Pin::as_mut(&mut fut).poll(&mut cx) {
        Poll::Ready(value) => Some(value),
        Poll::Pending => None,
    }
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

    #[test]
    fn apply_class_emits_chained_evidence_hash() {
        let arb = CouncilArbitrationEngine::new();
        let mut api = start_mercy_api_with_arbitration(None, &arb);
        let resp = api.handle_request(
            MercyApiRequest {
                kind: ApiRequestKind::CouncilQuery,
                payload: "tend the well and publish flow".into(),
                claimed_mercy: 0.99,
                actor: "operator".into(),
            },
            Some(&arb),
        );
        assert!(resp.accepted);
        let hash = resp.evidence_hash.expect("apply-class must emit a record");
        api.evidence_chain()
            .lock()
            .unwrap()
            .gate_side_effect(Some(&hash))
            .unwrap();
    }

    #[test]
    fn broken_chain_rejects_later_apply() {
        let arb = CouncilArbitrationEngine::new();
        let mut api = start_mercy_api_with_arbitration(None, &arb);
        let first = api.handle_request(
            MercyApiRequest {
                kind: ApiRequestKind::CouncilQuery,
                payload: "tend the well and publish flow".into(),
                claimed_mercy: 0.99,
                actor: "operator".into(),
            },
            Some(&arb),
        );
        assert!(first.accepted);
        api.evidence_chain()
            .lock()
            .unwrap()
            .tamper_last_previous_hash();
        let second = api.handle_request(
            MercyApiRequest {
                kind: ApiRequestKind::CouncilQuery,
                payload: "second apply without a sound chain".into(),
                claimed_mercy: 0.99,
                actor: "operator".into(),
            },
            Some(&arb),
        );
        assert!(!second.accepted);
        match second.decision {
            GateDecision::Rejected { reason } => {
                assert!(
                    reason.contains("evidence") || reason.contains("chain"),
                    "reason={reason}"
                );
            }
            GateDecision::Allowed => panic!("broken chain must not apply"),
        }
    }

    #[test]
    fn missing_record_is_not_apply() {
        let chain = EvidenceChain::new();
        assert!(chain.gate_side_effect(None).is_err());
        assert!(!crate::evidence_chain::apply_without_record_is_not_apply(
            false, true
        ));
    }
}
