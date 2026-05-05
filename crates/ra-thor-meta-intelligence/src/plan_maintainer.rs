//! Plan Maintainer
//!
//! This module gives Ra-Thor Meta-Intelligence the ability to read, analyze,
//! and suggest improvements to planning documents (PLANS.md, CORE_SPINE.md).
//! It is designed to grow into an active participant in maintaining and evolving
//! the project's strategic direction.

use serde::{Serialize, Deserialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanAnalysis {
    pub plans_md_health: f64,
    pub core_spine_alignment: f64,
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

    /// Analyze the current state of planning documents
    pub fn analyze_plans(&self) -> PlanAnalysis {
        let plans_content = self.read_file(&self.plans_path);
        let core_spine_content = self.read_file(&self.core_spine_path);

        // Simple initial heuristics (will be expanded significantly)
        let plans_health = if plans_content.contains("Integration Roadmap") { 0.85 } else { 0.65 };
        let spine_alignment = if core_spine_content.contains("Core Spine") { 0.90 } else { 0.70 };

        let mut suggestions = Vec::new();

        if !plans_content.contains("Status Tracking") {
            suggestions.push("Add more granular integration status tracking to PLANS.md".to_string());
        }
        if !core_spine_content.contains("Integration Priority") {
            suggestions.push("Add Integration Priority column to the Core Spine table".to_string());
        }

        PlanAnalysis {
            plans_md_health: plans_health,
            core_spine_alignment: spine_alignment,
            suggestions,
            last_analyzed: chrono::Utc::now().to_rfc3339(),
        }
    }

    fn read_file(&self, path: &str) -> String {
        fs::read_to_string(Path::new(path)).unwrap_or_default()
    }

    /// Future: This will generate improved versions of planning documents
    pub fn suggest_plan_updates(&self) -> String {
        "Meta-Intelligence: Generating suggested updates to planning documents... (In development)".to_string()
    }
}
