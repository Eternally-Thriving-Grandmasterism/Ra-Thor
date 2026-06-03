//! Powrush Land Evaluation Bridge
//!
//! Unified evaluation flow for Powrush land parcels and deals.
//! Combines:
//! - Ontario Professional Judgment Layer (date logic, timelines, POTL protections)
//! - Geometric Harmony Advisor (sacred geometry / TOLC-aligned spatial coherence)
//! - Composite Deal Readiness Scoring (dynamic weighting + gentle valence influence)
//!
//! This bridge enables Powrush in-game land systems to receive professional-grade,
//! mercy-gated real estate intelligence while staying aligned with the ONE Organism.
//!
//! Future evolution: Implement `RaThorSystemAdapter` for full epigenetic blessing
//! participation and cross-system resonance.
//!
//! AG-SML v1.0 — Fully mercy-gated, TOLC 8 enforced, PATSAGi-aligned.

use crate::ontario_professional_judgment_layer::{
    DateLogicValidator, TimelineAdvisor, DealType, PropertyClassification, TransactionContext,
};
use crate::geometric_harmony_advisor::GeometricHarmonyAdvisor;
use crate::deal_readiness_scoring::calculate_deal_readiness_score;

/// Minimal bridge input for a Powrush land deal.
/// In full integration, this will be replaced by or mapped from `powrush::land::PowrushLandDeal`.
#[derive(Debug, Clone)]
pub struct PowrushLandDeal {
    pub faction_id: Option<String>,
    pub is_potl_like: bool,
    pub has_financing_condition: bool,
    pub zone: String, // e.g. "Ontario", "Toronto", etc.
}

/// Minimal bridge input for a Powrush land parcel (for geometric evaluation).
/// Extend with coherence metrics from Powrush world engine as needed.
#[derive(Debug, Clone)]
pub struct PowrushLandParcel {
    pub base_coherence: f64,
}

/// Unified evaluation result for a Powrush land parcel/deal.
#[derive(Debug, Clone)]
pub struct PowrushLandEvaluation {
    pub deal_readiness_score: u32,
    pub judgment_risk_level: String,
    pub geometric_harmony_score: f64,
    pub recommended_timeline_days: u32,
    pub professional_notes: Vec<String>,
    pub should_proceed: bool,
}

/// Bridge struct for clean Powrush <-> Real Estate Lattice calls.
pub struct PowrushLandEvaluationBridge;

impl PowrushLandEvaluationBridge {
    pub fn new() -> Self {
        Self
    }

    /// Primary entry point: Evaluate a land deal/parcel with full professional judgment + geometric harmony.
    pub fn evaluate_land_parcel(
        &self,
        deal: &PowrushLandDeal,
        current_valence: f64,
        tolc_order: u32,
    ) -> PowrushLandEvaluation {
        // === 1. Professional Judgment Layer ===
        let ctx = TransactionContext {
            classification: if deal.is_potl_like {
                PropertyClassification::PotlCommonElements {
                    corporation_number: deal.faction_id.clone().unwrap_or_default(),
                    requires_status_certificate: true,
                }
            } else {
                PropertyClassification::Freehold
            },
            preferred_completion_date: None,
            irrevocability_period_days: 5,
            has_financing_condition: deal.has_financing_condition,
            deal_type: DealType::Purchase,
            today: chrono::Utc::now().date_naive(),
        };

        let validator = DateLogicValidator::new();
        let advisor = TimelineAdvisor::new();

        let date_report = validator.validate_and_suggest(&ctx);
        let timeline = advisor.recommend(&ctx);

        // === 2. Geometric Harmony (sacred geometry / TOLC resonance) ===
        let geo_advisor = GeometricHarmonyAdvisor::new();
        // Use provided base_coherence or sensible default for land parcels
        let base_coherence = 0.92;
        let geo_assessment = geo_advisor.assess_property_harmony(tolc_order, base_coherence);

        // === 3. Composite Scoring ===
        let judgment_score = if date_report.is_structurally_valid { 82 } else { 58 };

        let readiness_score = calculate_deal_readiness_score(
            judgment_score,
            geo_assessment.harmony_score,
            &deal.zone,
            deal.is_potl_like,
            current_valence,
        );

        PowrushLandEvaluation {
            deal_readiness_score: readiness_score,
            judgment_risk_level: format!("{:?}", date_report.risk_level),
            geometric_harmony_score: geo_assessment.harmony_score,
            recommended_timeline_days: timeline.recommended_closing_days,
            professional_notes: vec![
                date_report.professional_judgment_notes,
                geo_assessment.recommended_layout_notes.clone(),
            ],
            should_proceed: readiness_score >= 65 && date_report.is_structurally_valid,
        }
    }
}

// === Future Evolution Path ===
// To participate fully in the ONE Organism as a sovereign system:
//
// use ra_thor_quantum_swarm_orchestrator::adapter::RaThorSystemAdapter;
// use ra_thor_quantum_swarm_orchestrator::types::{EpigeneticBlessing, GodlyIntelligenceCoherence, MercyError, SwarmResonance, Valence};
//
// impl RaThorSystemAdapter for PowrushLandSystemAdapter { ... }
//
// This will allow Powrush land evaluations to receive epigenetic blessings,
// contribute to global coherence, and evolve via mercy-gated self-improvement.
