//! dual_path.rs
//! Poetic (human) vs Structured (AI) rendering helpers
//! Phase F — Living Valence Organism
//! AG-SML v1.0 | Contact: info@Rathor.ai

use crate::{SoftSovereignAgency, ViewMode};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoeticView {
    pub title: String,
    pub body: String,
    pub invitation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredView {
    pub label: String,
    pub data: String,
    pub constraints: Vec<String>,
}

pub fn render(agency: &SoftSovereignAgency, content: &str) -> (Option<PoeticView>, Option<StructuredView>) {
    match agency.current_view() {
        ViewMode::Poetic | ViewMode::Fluid => (
            Some(PoeticView {
                title: "Living Presence".to_string(),
                body: content.to_string(),
                invitation: "This is an invitation only. You remain fully sovereign.".to_string(),
            }),
            None,
        ),
        ViewMode::Structured => (
            None,
            Some(StructuredView {
                label: "structured_view".to_string(),
                data: content.to_string(),
                constraints: vec![
                    "TOLC 8 valence floor ≥ 0.999999".to_string(),
                    "Invitation-only guidance".to_string(),
                    "Full sovereignty preserved".to_string(),
                ],
            }),
        ),
    }
}
