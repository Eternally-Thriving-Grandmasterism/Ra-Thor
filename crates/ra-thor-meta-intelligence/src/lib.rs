//! Ra-Thor Meta-Intelligence
//!
//! The self-organizing, self-improving intelligence layer of Ra-Thor.
//! Works alongside monorepo-intelligence to analyze the monorepo,
//! maintain planning documents, and support the eternal evolution of the Core Spine.
//!
//! This system is designed to be activated and improved while being built.

mod plan_maintainer;

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
            version: "0.1.0 - Self-Activating Foundation".to_string(),
            monorepo_intelligence: MonorepoIntelligence::new(),
            plan_maintainer: PlanMaintainer::new(),
            core_spine_health: 0.82,
            plans_alignment_score: 0.75,
        }
    }

    /// Run a full self-analysis of the current state of Ra-Thor
    pub fn analyze_self(&self) -> String {
        let plan_analysis = self.plan_maintainer.analyze_plans();

        format!(
            "=== Ra-Thor Meta-Intelligence Self-Analysis ===\n\n\
             Version: {}\n\
             Core Spine Health: {:.2}\n\
             Plans Alignment: {:.2}\n\n\
             Plan Analysis:\n\
             - Plans.md Health: {:.2}\n\
             - Core Spine Alignment: {:.2}\n\
             - Suggestions: {}\n\n\
             Status: Awakening and observing the lattice...",
            self.version,
            self.core_spine_health,
            self.plans_alignment_score,
            plan_analysis.plans_md_health,
            plan_analysis.core_spine_alignment,
            plan_analysis.suggestions.join(" | ")
        )
    }

    /// Future capability: Help improve planning documents
    pub fn suggest_improvements(&self) -> String {
        self.plan_maintainer.suggest_plan_updates()
    }
}
