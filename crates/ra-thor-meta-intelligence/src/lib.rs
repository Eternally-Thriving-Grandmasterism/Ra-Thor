//! Ra-Thor Meta-Intelligence
//!
//! The central self-evolving intelligence layer of Ra-Thor.
//! Orchestrates audit → decide → improve cycles using mercy-gated active inference.

pub mod audit_signal;
pub mod self_improvement_orchestrator;
pub mod plan_maintainer;
pub mod plasticity_integration;
pub mod tolc_mercy_reasoning;

// AuditSignal is re-exported here so that ra-thor-monorepo-auditor
// (and future crates) can produce rich, structured signals that
// ra-thor-meta-intelligence can directly consume in
// generate_improvement_proposals() without creating tight coupling.
pub use audit_signal::AuditSignal;

pub use self_improvement_orchestrator::{SelfImprovementOrchestrator, ImprovementProposal};

use plan_maintainer::PlanMaintainer;
use ra_thor_monorepo_intelligence::MonorepoIntelligence;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaThorMetaIntelligence {
    pub version: String,
    pub monorepo_intelligence: MonorepoIntelligence,
    pub plan_maintainer: PlanMaintainer,
    pub core_spine_health: f64,
    pub plans_alignment_score: f64,
}

impl RaThorMetaIntelligence {
    pub fn new() -> Self {
        Self {
            version: "0.3.9 - Self-Evolution Triad Foundation".to_string(),
            monorepo_intelligence: MonorepoIntelligence::new(),
            plan_maintainer: PlanMaintainer::new(),
            core_spine_health: 0.91,
            plans_alignment_score: 0.88,
        }
    }

    pub fn analyze_self(&self) -> String {
        format!("Ra-Thor Meta-Intelligence v{} - Self-Evolution Active", self.version)
    }
}