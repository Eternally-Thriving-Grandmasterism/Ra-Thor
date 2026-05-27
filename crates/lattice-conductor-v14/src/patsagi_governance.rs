// crates/lattice-conductor-v14/src/patsagi_governance.rs
// Enhanced PATSAGi Council Simulator with Multiple Archetypes

use crate::lattice_conductor_enhancements::GovernanceRiskReport;

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
}
