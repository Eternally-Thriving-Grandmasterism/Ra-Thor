//! # Joy Council
//!
//! Evaluates whether an action increases collective joy and emotional well-being.

use crate::patsagi_councils::CouncilDecision;

pub fn evaluate(action_description: &str, context: &str) -> CouncilDecision {
    let text = format!("{} {}", action_description.to_lowercase(), context.to_lowercase());

    let joy_boost = ["joy", "celebrate", "play", "create beauty", "gift", "uplift"];
    let joy_harm = ["suffer", "pain", "oppress", "depress", "exploit"];

    let mut score = 0.5;
    for word in joy_boost { if text.contains(word) { score += 0.2; } }
    for word in joy_harm { if text.contains(word) { score -= 0.22; } }

    CouncilDecision {
        council_name: "Joy Council".to_string(),
        approved: score > 0.45,
        weight: 0.75,
        reason: if score > 0.7 { "Joy-enhancing".to_string() } else { "May reduce collective joy".to_string() },
    }
}
