//! # Harmony Council
//!
//! Evaluates whether an action increases or decreases systemic harmony.

use crate::patsagi_councils::CouncilDecision;

pub fn evaluate(action_description: &str, context: &str) -> CouncilDecision {
    let text = format!("{} {}", action_description.to_lowercase(), context.to_lowercase());

    let disruptive = ["conflict", "divide", "oppose", "attack", "destroy"];
    let harmonizing = ["unite", "heal", "balance", "integrate", "peace"];

    let mut score = 0.5;
    for word in disruptive { if text.contains(word) { score -= 0.2; } }
    for word in harmonizing { if text.contains(word) { score += 0.18; } }

    CouncilDecision {
        council_name: "Harmony Council".to_string(),
        approved: score > 0.4,
        weight: 0.8,
        reason: if score > 0.65 { "Promotes harmony".to_string() } else { "May create disharmony".to_string() },
    }
}
