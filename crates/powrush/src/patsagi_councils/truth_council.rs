//! # Truth Council
//!
//! Evaluates honesty, transparency, and alignment with verifiable reality.

use crate::patsagi_councils::CouncilDecision;

pub fn evaluate(action_description: &str, context: &str) -> CouncilDecision {
    let text = format!("{} {}", action_description.to_lowercase(), context.to_lowercase());

    let deceptive = ["lie", "deceive", "hide", "mislead", "obscure"];
    let truthful = ["reveal", "share truth", "transparent", "honest", "verify"];

    let mut score = 0.5;
    for word in deceptive { if text.contains(word) { score -= 0.25; } }
    for word in truthful { if text.contains(word) { score += 0.2; } }

    CouncilDecision {
        council_name: "Truth Council".to_string(),
        approved: score > 0.35,
        weight: 0.9,
        reason: if score > 0.6 { "Strong truth alignment".to_string() } else { "Potential deception risk".to_string() },
    }
}
