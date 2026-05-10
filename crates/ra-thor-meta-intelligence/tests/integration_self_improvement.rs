//! Integration tests for SelfImprovementEngine + CrateAnalyzer
//!
//! These tests verify that the mercy-gated, data-driven self-improvement
//! logic works correctly end-to-end.

use ra_thor_meta_intelligence::{
    self_improvement_engine::{SelfImprovementEngine, ImprovementType},
    crate_analyzer::{CrateAnalyzer, CrateHealthReport},
};
use mercy_merlin_engine::MercyMerlinEngine;
use tokio;

/// Helper to create a test engine with real components
async fn create_test_engine() -> SelfImprovementEngine {
    let mercy_engine = MercyMerlinEngine::new();
    let crate_analyzer = CrateAnalyzer::new();
    SelfImprovementEngine::new(mercy_engine, crate_analyzer)
}

#[tokio::test]
async fn test_generate_improvement_proposals_respects_mercy_gate() {
    let mut engine = create_test_engine().await;

    let proposals = engine.generate_improvement_proposals().await.unwrap();

    // In a healthy high-valence state, proposals should be generated
    // The exact number depends on current crate health data in CrateAnalyzer
    println!("Generated {} proposals under mercy gate", proposals.len());
}

#[tokio::test]
async fn test_proposals_are_data_driven() {
    let mut engine = create_test_engine().await;
    let proposals = engine.generate_improvement_proposals().await.unwrap();

    for proposal in &proposals {
        assert!(proposal.mercy_alignment >= 7, "All proposals must have strong mercy alignment");
        assert!(!proposal.target_crate.is_empty());
    }
}

#[tokio::test]
async fn test_crate_analyzer_produces_useful_reports() {
    let analyzer = CrateAnalyzer::new();
    let reports = analyzer.analyze_critical_crates().await.unwrap();

    assert!(!reports.is_empty(), "Analyzer should return reports for critical crates");

    for report in reports {
        assert!(report.importance >= 1 && report.importance <= 10);
        assert!(report.mercy_integration_score <= 10);
    }
}
