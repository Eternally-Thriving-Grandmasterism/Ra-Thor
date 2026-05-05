//! Plan Maintainer
//!
//! Gives Ra-Thor Meta-Intelligence the ability to analyze and suggest
//! improvements to planning documents (PLANS.md, CORE_SPINE.md).
//! Designed to grow into an active participant in strategic maintenance.

use serde::{Serialize, Deserialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanAnalysis {
    pub plans_md_health: f64,
    pub core_spine_alignment: f64,
    pub integration_status_score: f64,
    pub suggestions: Vec<String>,
    pub last_analyzed: String,
}

pub struct PlanMaintainer {
    pub plans_path: String,
    pub core_spine_path: String,
}

impl PlanMaintainer {
    pub fn new() -> Self {
        Self {
            plans_path: "PLANS.md".to_string(),
            core_spine_path: "CORE_SPINE.md".to_string(),
        }
    }

    /// Comprehensive analysis of current planning documents
    pub fn analyze_plans(&self) -> PlanAnalysis {
        let plans_content = self.read_file(&self.plans_path);
        let core_spine_content = self.read_file(&self.core_spine_path);

        let plans_health = if plans_content.contains("Integration Roadmap") { 0.88 } else { 0.65 };
        let spine_alignment = if core_spine_content.contains("Core Spine") { 0.92 } else { 0.70 };
        let integration_score = if plans_content.contains("Status Tracking") { 0.80 } else { 0.55 };

        let mut suggestions = Vec::new();

        if !plans_content.contains("Status Tracking") {
            suggestions.push("Add granular integration status tracking table to PLANS.md".to_string());
        }
        if !core_spine_content.contains("Integration Priority") {
            suggestions.push("Add Integration Priority column to the Core Spine table".to_string());
        }
        if !plans_content.contains("Crate Organization Strategy") {
            suggestions.push("Strengthen the Crate Organization Strategy section".to_string());
        }

        PlanAnalysis {
            plans_md_health: plans_health,
            core_spine_alignment: spine_alignment,
            integration_status_score: integration_score,
            suggestions,
            last_analyzed: chrono::Utc::now().to_rfc3339(),
        }
    }

    fn read_file(&self, path: &str) -> String {
        fs::read_to_string(Path::new(path)).unwrap_or_default()
    }

    pub fn suggest_plan_updates(&self) -> String {
        "Meta-Intelligence: Generating suggested updates to planning documents... (Under active development)".to_string()
    }
}
