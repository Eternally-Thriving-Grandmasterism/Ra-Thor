// crates/lattice-conductor-v14/src/patsagi_governance.rs
// Enhanced PATSAGi Council Simulator with Multiple Archetypes

use crate::lattice_conductor_enhancements::GovernanceRiskReport;
use crate::evidence_chain::{EvidenceChain, EvidenceDraft, EvidenceError, EvidenceKind};
use crate::lipschitz_gate::{LipschitzCheck, LipschitzError, LipschitzGate};

#[derive(Debug, Clone)]
pub struct PatsagiReviewRequest {
    pub topic: String,
    pub summary: String,
    pub mercy_impact_score: f64,
    pub requested_by: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PatsagiDecision {
    Approved { confidence: f64 },
    RequiresSelfEvolution { priority: u8 },
    RequiresCouncilArbitration { councils: Vec<u32> },
    Rejected { reason: String, mercy_impact: f64 },
}

/// PATSAGi Council Simulator with support for multiple archetypes
pub struct PatsagiCouncilSimulator;

impl PatsagiCouncilSimulator {
    /// Default review (balanced)
    pub fn review(request: &PatsagiReviewRequest) -> PatsagiDecision {
        if request.mercy_impact_score < 0.75 {
            PatsagiDecision::RequiresSelfEvolution { priority: 2 }
        } else {
            PatsagiDecision::Approved { confidence: 0.85 }
        }
    }

    /// Mercy-focused council review
    pub fn review_as_mercy_council(request: &PatsagiReviewRequest, risk: Option<&GovernanceRiskReport>) -> PatsagiDecision {
        if let Some(r) = risk {
            if r.risk_score > 0.82 {
                return PatsagiDecision::RequiresSelfEvolution { priority: 2 };
            }
        }
        PatsagiDecision::Approved { confidence: 0.82 }
    }

    /// Truth-focused council review
    pub fn review_as_truth_council(request: &PatsagiReviewRequest, risk: Option<&GovernanceRiskReport>) -> PatsagiDecision {
        if let Some(r) = risk {
            if r.max_banzhaf > 0.65 {
                return PatsagiDecision::RequiresSelfEvolution { priority: 3 };
            }
        }
        PatsagiDecision::Approved { confidence: 0.88 }
    }

    /// Council #13 (Supreme Architect) - strictest review
    pub fn review_as_council_13(request: &PatsagiReviewRequest, risk: Option<&GovernanceRiskReport>) -> PatsagiDecision {
        if let Some(r) = risk {
            if r.max_banzhaf > 0.60 || r.risk_score > 0.70 {
                return PatsagiDecision::RequiresSelfEvolution { priority: 4 };
            }
        }
        PatsagiDecision::Approved { confidence: 0.93 }
    }

    /// Evidential face for a council row. Same actor on proposal and audit is Rejected.
    /// Does not flip a prior Layer 0 / Lipschitz Reject into Apply.
    pub fn review_with_evidence(
        request: &PatsagiReviewRequest,
        chain: &mut EvidenceChain,
        auditor: &str,
        timestamp: u64,
    ) -> Result<(PatsagiDecision, String), EvidenceError> {
        if auditor.trim().is_empty() || auditor == request.requested_by {
            return Err(EvidenceError::SelfAudit {
                actor: request.requested_by.clone(),
            });
        }
        let decision = Self::review(request);
        let (directive, claim) = match &decision {
            PatsagiDecision::Approved { confidence } => {
                ("accept".to_string(), format!("council-approved confidence={confidence}"))
            }
            PatsagiDecision::Rejected { reason, .. } => {
                ("reject".to_string(), format!("council-rejected {reason}"))
            }
            PatsagiDecision::RequiresSelfEvolution { priority } => {
                ("refine".to_string(), format!("requires-self-evolution priority={priority}"))
            }
            PatsagiDecision::RequiresCouncilArbitration { .. } => {
                ("refine".to_string(), "requires-council-arbitration".into())
            }
        };
        let rec = chain.append(EvidenceDraft {
            subject: format!("council:{}", request.requested_by),
            input: crate::evidence_chain::payload_digest(&request.summary),
            claim,
            evidence_pointers: vec!["patsagi-review".into(), "layer0-first".into()],
            directive: directive.clone(),
            scope: "lattice-conductor-v14".into(),
            kind: EvidenceKind::Council,
            decision: directive,
            timestamp,
            actor: request.requested_by.clone(),
            auditor: auditor.into(),
        })?;
        Ok((decision, rec.hash))
    }

    /// Freeze a new Lipschitz ball only after an accepted check. Cannot enlarge `r`.
    pub fn freeze_ball_after_approve(
        gate: &mut LipschitzGate,
        check: &LipschitzCheck,
        proposed_r: Option<f64>,
    ) -> Result<(), LipschitzError> {
        if let Some(r) = proposed_r {
            gate.council_may_not_enlarge_radius(r)?;
        }
        gate.freeze_if_accepted(check)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_chain::EvidenceChain;
    use crate::lipschitz_gate::{LipschitzBall, LipschitzGate, Theta};

    #[test]
    fn council_self_audit_is_rejected() {
        let mut chain = EvidenceChain::new();
        let request = PatsagiReviewRequest {
            topic: "flip a rejected gate".into(),
            summary: "majority wants apply anyway".into(),
            mercy_impact_score: 0.99,
            requested_by: "council-13".into(),
        };
        let err = PatsagiCouncilSimulator::review_with_evidence(
            &request, &mut chain, "council-13", 1,
        )
        .unwrap_err();
        assert!(matches!(err, EvidenceError::SelfAudit { .. }));
        assert!(chain.records().is_empty());
    }

    #[test]
    fn distinct_auditor_emits_council_row() {
        let mut chain = EvidenceChain::new();
        let request = PatsagiReviewRequest {
            topic: "harden wrap".into(),
            summary: "tend the well".into(),
            mercy_impact_score: 0.99,
            requested_by: "operator".into(),
        };
        let (decision, hash) = PatsagiCouncilSimulator::review_with_evidence(
            &request,
            &mut chain,
            "council-13",
            2,
        )
        .unwrap();
        assert!(matches!(decision, PatsagiDecision::Approved { .. }));
        chain.gate_side_effect(Some(&hash)).unwrap();
    }

    #[test]
    fn council_cannot_freeze_rejected_or_enlarge_r() {
        let mut gate = LipschitzGate::new();
        gate.install(LipschitzBall::try_new(Theta(vec![0.0, 0.0]), 1.0, 2.0).unwrap())
            .unwrap();
        let outside = gate.verify(&Theta(vec![0.9, 0.0]));
        assert!(!outside.accepted);
        assert!(PatsagiCouncilSimulator::freeze_ball_after_approve(
            &mut gate,
            &outside,
            Some(10.0)
        )
        .is_err());

        let inside = gate.verify(&Theta(vec![0.2, 0.0]));
        assert!(inside.accepted);
        assert!(PatsagiCouncilSimulator::freeze_ball_after_approve(
            &mut gate,
            &inside,
            Some(10.0)
        )
        .is_err());
        PatsagiCouncilSimulator::freeze_ball_after_approve(&mut gate, &inside, None).unwrap();
        assert_eq!(gate.generation(), 2);
    }
}
