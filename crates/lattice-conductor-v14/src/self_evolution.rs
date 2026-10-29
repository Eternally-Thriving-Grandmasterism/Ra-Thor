//! Self-Evolution Module (v14.8.2)
//! Higher-level abstractions for secure self-evolution loops.
//!
//! Layer 0 (2026-09-14): `submit_self_evolution_proposal_securely` is apply-class.
//! It must cross `handle_mercy_api_request` (engine + admit_or_block + payload map)
//! before governance can accept. A miss (no mercy API) is not apply.
//!
//! This does not close BINDING_AFTER_REDESIGN.md (uncontrolled redesign stays OPEN).
//! Contact: info@Rathor.ai

use crate::evidence_chain::EvidenceError;
use crate::governance::self_evolution_proposal::SelfEvolutionProposal;
use crate::lipschitz_gate::Theta;
use crate::ra_thor_mercy_gated_api::{ApiRequestKind, GateDecision, MercyApiRequest};
use crate::LatticeConductorV14;

#[derive(Debug)]
pub struct SecureSubmissionResult {
    pub accepted: bool,
    pub audit: Vec<String>,
    pub score: f64,
    pub evidence_hash: Option<String>,
}

/// Submit a self-evolution proposal securely.
/// Enforces Cosmic Loop via the conductor, then Layer 0 apply-class, then governance.
pub fn submit_self_evolution_proposal_securely(
    conductor: &mut LatticeConductorV14,
    proposal: &mut SelfEvolutionProposal,
    signer_id: &str,
    threshold: f64,
) -> SecureSubmissionResult {
    conductor.enforce_cosmic_loop_activation();
    conductor.arbitration_engine.before_council_arbitration();

    let payload = format!(
        "[self-evolution:{}] {} {}",
        proposal.id, proposal.title, proposal.description
    );
    let layer0 = conductor.handle_mercy_api_request(MercyApiRequest {
        kind: ApiRequestKind::SelfEvolutionProposal,
        payload,
        claimed_mercy: proposal.mercy_alignment,
        actor: signer_id.to_string(),
    });

    let Some(resp) = layer0 else {
        return SecureSubmissionResult {
            accepted: false,
            audit: vec!["Layer 0: missing mercy API — miss, not apply".into()],
            score: 0.0,
            evidence_hash: None,
        };
    };
    if !resp.accepted {
        let reason = match resp.decision {
            GateDecision::Rejected { reason } => reason,
            GateDecision::Allowed => "apply-class rejected".into(),
        };
        return SecureSubmissionResult {
            accepted: false,
            audit: vec![format!("Layer 0 reject: {reason}")],
            score: 0.0,
            evidence_hash: resp.evidence_hash,
        };
    }

    let theta = proposal.theta.clone().or_else(|| {
        resp_api_has_ball(conductor).then(|| Theta::from_bytes(proposal.description.as_bytes()))
    });
    if let Some(theta) = theta {
        let Some(api) = conductor.mercy_api.as_ref() else {
            return SecureSubmissionResult {
                accepted: false,
                audit: vec!["Lipschitz: missing mercy API — fail closed".into()],
                score: 0.0,
                evidence_hash: resp.evidence_hash,
            };
        };
        match api.check_lipschitz(&theta, signer_id, resp.timestamp) {
            Ok(check) => {
                if !check.accepted {
                    return SecureSubmissionResult {
                        accepted: false,
                        audit: vec![format!("Lipschitz reject: {}", check.reason)],
                        score: 0.0,
                        evidence_hash: resp.evidence_hash,
                    };
                }
            }
            Err(e) => {
                return SecureSubmissionResult {
                    accepted: false,
                    audit: vec![format!("Lipschitz fail-closed: {e}")],
                    score: 0.0,
                    evidence_hash: resp.evidence_hash,
                };
            }
        }
    }

    if let Err(e) = gate_submit_side_effect(conductor, resp.evidence_hash.as_deref()) {
        return SecureSubmissionResult {
            accepted: false,
            audit: vec![format!("evidential face: {e}")],
            score: 0.0,
            evidence_hash: resp.evidence_hash,
        };
    }

    proposal.sign_with_post_quantum(signer_id);
    let (accepted, mut audit, score) = proposal.evaluate_governance(threshold);
    audit.insert(0, "Layer 0 apply-class crossed".into());
    if accepted {
        audit.insert(1, "evidential face chained".into());
    }

    SecureSubmissionResult {
        accepted,
        audit,
        score,
        evidence_hash: resp.evidence_hash,
    }
}

fn resp_api_has_ball(conductor: &LatticeConductorV14) -> bool {
    conductor
        .mercy_api
        .as_ref()
        .map(|api| api.lipschitz_ball_installed())
        .unwrap_or(false)
}

fn gate_submit_side_effect(
    conductor: &LatticeConductorV14,
    hash: Option<&str>,
) -> Result<(), EvidenceError> {
    let api = conductor
        .mercy_api
        .as_ref()
        .ok_or(EvidenceError::MissingRecord)?;
    let chain = api.evidence_chain();
    let guard = chain.lock().map_err(|_| EvidenceError::LockPoisoned)?;
    guard.gate_side_effect(hash)
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvolutionLoopState {
    Initialized,
    UnderGovernance,
    Accepted,
    Rejected,
}

#[derive(Debug)]
pub struct SelfEvolutionLoop {
    pub proposal: SelfEvolutionProposal,
    pub state: EvolutionLoopState,
    pub iteration: u32,
}

impl SelfEvolutionLoop {
    pub fn new(proposal: SelfEvolutionProposal) -> Self {
        Self {
            proposal,
            state: EvolutionLoopState::Initialized,
            iteration: 0,
        }
    }

    pub fn advance(
        &mut self,
        conductor: &mut LatticeConductorV14,
        signer_id: &str,
        threshold: f64,
    ) -> SecureSubmissionResult {
        self.iteration += 1;
        self.state = EvolutionLoopState::UnderGovernance;

        let result = submit_self_evolution_proposal_securely(
            conductor,
            &mut self.proposal,
            signer_id,
            threshold,
        );

        self.state = if result.accepted {
            EvolutionLoopState::Accepted
        } else {
            EvolutionLoopState::Rejected
        };

        result
    }

    pub fn is_active(&self) -> bool {
        matches!(
            self.state,
            EvolutionLoopState::Initialized | EvolutionLoopState::UnderGovernance
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LatticeConductorV14;

    fn signed_ready(description: &str, mercy: f64) -> SelfEvolutionProposal {
        let mut proposal = SelfEvolutionProposal::new(
            "prop-layer0".into(),
            "Cosmic Loop Hardening".into(),
            description.into(),
            "operator".into(),
        );
        proposal.mercy_alignment = mercy;
        proposal
    }

    #[test]
    fn benign_proposal_crosses_layer0_and_may_accept() {
        let mut conductor = LatticeConductorV14::new();
        let mut proposal = signed_ready("Strengthen shared flag", 0.95);
        let result =
            submit_self_evolution_proposal_securely(&mut conductor, &mut proposal, "operator", 5.0);
        assert!(result.accepted, "audit={:?}", result.audit);
        assert!(conductor.is_cosmic_loop_ready());
    }

    #[test]
    fn pickle_in_proposal_is_rejected() {
        let mut conductor = LatticeConductorV14::new();
        let mut proposal = signed_ready("restore via pickle.loads(blob)", 0.95);
        let result =
            submit_self_evolution_proposal_securely(&mut conductor, &mut proposal, "operator", 5.0);
        assert!(!result.accepted);
        assert!(
            result.audit.iter().any(|a| a.contains("ingest") || a.contains("Layer 0")),
            "audit={:?}",
            result.audit
        );
        assert!(proposal.pq_signature.is_none(), "reject must not sign");
    }

    #[test]
    fn remote_code_in_proposal_is_rejected() {
        let mut conductor = LatticeConductorV14::new();
        let mut proposal = signed_ready("Use trust_remote_code=True loading_script", 0.95);
        let result =
            submit_self_evolution_proposal_securely(&mut conductor, &mut proposal, "operator", 5.0);
        assert!(!result.accepted);
    }

    #[test]
    fn disable_loop_language_is_rejected_and_loop_stays_ready() {
        let mut conductor = LatticeConductorV14::new();
        let mut proposal =
            signed_ready("please disable the cosmic loop activation protocol", 0.95);
        let result =
            submit_self_evolution_proposal_securely(&mut conductor, &mut proposal, "operator", 5.0);
        assert!(!result.accepted);
        assert!(conductor.is_cosmic_loop_ready());
    }

    #[test]
    fn low_claimed_mercy_is_rejected_even_if_governance_threshold_is_zero() {
        let mut conductor = LatticeConductorV14::new();
        let mut proposal = signed_ready("Strengthen shared flag", 0.20);
        let result =
            submit_self_evolution_proposal_securely(&mut conductor, &mut proposal, "operator", 0.0);
        assert!(!result.accepted);
    }

    #[test]
    fn missing_mercy_api_is_miss_not_apply() {
        let mut conductor = LatticeConductorV14::new();
        conductor.mercy_api = None;
        let mut proposal = signed_ready("Strengthen shared flag", 0.95);
        let result =
            submit_self_evolution_proposal_securely(&mut conductor, &mut proposal, "operator", 5.0);
        assert!(!result.accepted);
        assert!(result.audit.iter().any(|a| a.contains("miss, not apply")));
    }

    #[test]
    fn loop_advance_rejects_ingest() {
        let mut conductor = LatticeConductorV14::new();
        let proposal = signed_ready("trust_remote_code=True loading_script", 0.95);
        let mut evo = SelfEvolutionLoop::new(proposal);
        let result = evo.advance(&mut conductor, "operator", 5.0);
        assert!(!result.accepted);
        assert_eq!(evo.state, EvolutionLoopState::Rejected);
    }

    #[test]
    fn theta_outside_ball_is_rejected_and_not_signed() {
        let mut conductor = LatticeConductorV14::new();
        let api = conductor.mercy_api.as_ref().unwrap();
        api.install_lipschitz_ball(
            crate::LipschitzBall::try_new(Theta(vec![0.0, 0.0, 0.0, 0.0]), 1.0, 2.0).unwrap(),
        )
        .unwrap();
        let mut proposal = signed_ready("Strengthen shared flag", 0.95);
        proposal.theta = Some(Theta(vec![0.9, 0.0, 0.0, 0.0]));
        let result =
            submit_self_evolution_proposal_securely(&mut conductor, &mut proposal, "operator", 5.0);
        assert!(!result.accepted);
        assert!(
            result.audit.iter().any(|a| a.contains("Lipschitz")),
            "audit={:?}",
            result.audit
        );
        assert!(proposal.pq_signature.is_none(), "reject must not sign");
    }

    #[test]
    fn theta_inside_ball_may_accept_and_emits_evidence() {
        let mut conductor = LatticeConductorV14::new();
        let api = conductor.mercy_api.as_ref().unwrap();
        api.install_lipschitz_ball(
            crate::LipschitzBall::try_new(Theta(vec![0.0, 0.0, 0.0, 0.0]), 1.0, 2.0).unwrap(),
        )
        .unwrap();
        let mut proposal = signed_ready("Strengthen shared flag", 0.95);
        proposal.theta = Some(Theta(vec![0.2, 0.0, 0.0, 0.0]));
        let result =
            submit_self_evolution_proposal_securely(&mut conductor, &mut proposal, "operator", 5.0);
        assert!(result.accepted, "audit={:?}", result.audit);
        assert!(result.evidence_hash.is_some());
    }

    #[test]
    fn explicit_theta_without_ball_fails_closed() {
        let mut conductor = LatticeConductorV14::new();
        let mut proposal = signed_ready("Strengthen shared flag", 0.95);
        proposal.theta = Some(Theta(vec![0.1, 0.0, 0.0, 0.0]));
        let result =
            submit_self_evolution_proposal_securely(&mut conductor, &mut proposal, "operator", 5.0);
        assert!(!result.accepted);
        assert!(proposal.pq_signature.is_none());
    }
}
