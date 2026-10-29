//! # Abundance Council
//!
//! Evaluates whether an action increases or decreases systemic abundance.
//! Strongly favors post-scarcity outcomes and resource regeneration.

use crate::patsagi_councils::CouncilDecision;

pub fn evaluate(action_description: &str, context: &str) -> CouncilDecision {
    let text = format!("{} {}", action_description.to_lowercase(), context.to_lowercase());

    let positive = ["create", "regenerate", "abundance", "share", "gift", "build", "heal"];
    let negative = ["hoard", "restrict", "monopolize", "deplete", "waste"];

    let mut score = 0.5;

    for word in positive {
        if text.contains(word) { score += 0.15; }
    }
    for word in negative {
        if text.contains(word) { score -= 0.2; }
    }

    CouncilDecision {
        council_name: "Abundance Council".to_string(),
        approved: score > 0.4,
        weight: 0.85,
        reason: if score > 0.6 {
            "Strongly supports abundance".to_string()
        } else {
            "May reduce systemic abundance".to_string()
        },
    }
}
