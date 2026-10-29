//! Integration test: Full LegacyCompatibilityBridge cycle
use ra_thor::legacy_compatibility_bridge::LegacyCompatibilityBridge;

#[test]
fn test_legacy_compatibility_bridge_full_cycle() {
    let bridge = LegacyCompatibilityBridge::new();

    let old_proposal = "Old pre-2025 mercy system";
    let adapted = bridge.adapt_legacy_self_evolution_loop(old_proposal, "v4");

    assert!(adapted.adapted_feedback.contains("Legacy-adapted v1.1"));
    assert!(adapted.validation.contains("PASSED"));
}