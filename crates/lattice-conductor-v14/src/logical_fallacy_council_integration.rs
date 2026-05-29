//! Integration: Logical Fallacy Detection wired into PATSAGi Council guidance
// Thunder Lattice v14.2.2 — Mercy-gated truth-seeking layer

use crate::logical_fallacy_detection::{LogicalFallacyDetector, FallacyType};
use crate::healing_integration::HealingFieldRegistry;

/// Runs fallacy detection on council deliberation text or healing rationale.
/// Flags fallacies before mercy propagation or proposal acceptance.
pub fn apply_fallacy_guard_to_council_guidance(
    registry: &mut HealingFieldRegistry,
    chat_id: &str,
    deliberation_text: &str,
) -> Vec<FallacyType> {
    let detector = LogicalFallacyDetector::new();
    let detected = detector.detect_fallacies(deliberation_text);
    
    if !detected.is_empty() {
        // In real impl: lower mercy_flow or require council re-deliberation
        if let Some(field) = registry.get_field_mut(chat_id) {
            field.mercy_flow *= 0.7; // Mercy penalty for fallacious reasoning
        }
    }
    detected.into_iter().map(|f| f.fallacy_type).collect()
}

// This is now called inside apply_patsagi_council_guidance and simulate_healing_step
// Beautiful truth-seeking upgrade for all council deliberations.