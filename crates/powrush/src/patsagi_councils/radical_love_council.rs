//! # Radical Love Council (Supreme Veto Council)
//!
//! The **first and highest** of the PATSAGi Councils.
//! Any action that fails Radical Love evaluation is immediately rejected,
//! regardless of other council opinions.

use crate::patsagi_councils::CouncilDecision;

pub fn evaluate(action_description: &str, context: &str) -> CouncilDecision {
    let text = format!("{} {}", action_description.to_lowercase(), context.to_lowercase());

    // Hard veto patterns (non-negotiable)
    let harmful = ["harm", "exploit", "deceive", "dominate", "kill", "destroy", "coerce"];

    for pattern in harmful {
        if text.contains(pattern) {
            return CouncilDecision {
                council_name: "Radical Love Council".to_string(),
                approved: false,
                weight: 1.0, // Supreme weight
                reason: format!("Vetoed: harmful pattern '{}' detected", pattern),
            };
        }
    }

    // Default approval with high weight
    CouncilDecision {
        council_name: "Radical Love Council".to_string(),
        approved: true,
        weight: 1.0,
        reason: "Approved — no harmful intent detected".to_string(),
    }
}
