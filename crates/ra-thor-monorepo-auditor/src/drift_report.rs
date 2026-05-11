use serde::{Deserialize, Serialize};

/// Report on detected drift, outdated patterns, and inconsistencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    pub outdated_cargo_toml_count: u32,
    pub broken_dependency_links: u32,
    pub missing_mercy_gating: u32,
    pub hallucination_risk_areas: Vec<String>,
    pub recommendations: Vec<String>,
}

impl DriftReport {
    pub fn new() -> Self {
        Self {
            outdated_cargo_toml_count: 0,
            broken_dependency_links: 0,
            missing_mercy_gating: 0,
            hallucination_risk_areas: vec![],
            recommendations: vec![],
        }
    }
}