//! Valuation Confidence Scorer for Real Estate Lattice
//!
//! Generates context-aware valuation confidence scores and merciful explanations
//! by combining signals from PropertyType, DealType, StatusCertificateAnalysis,
//! DeveloperRisk, and Multi-Offer dynamics.
//!
//! This is the foundation for a future Hybrid AVM that is more accurate and
//! protective than pure data-driven models because it incorporates real-time
//! risk and market pressure signals that traditional AVMs miss.
//!
//! **Design Principles**:
//! - Confidence scoring instead of overconfident point estimates
//! - Merciful, explainable output (not just a number)
//! - Strong weighting on risk factors our lattice already tracks well
//! - PATSAGi / ethical awareness (e.g. family transfers, high escalation)
//!
//! Part of the evolving Hybrid Valuation capability in the Real Estate Lattice.

use crate::property_type_classifier::OntarioPropertyType;
use crate::deal_type_classifier::DealType;
use crate::status_certificate_analyzer::StatusCertificateAnalysis;
use crate::developer_risk_engine::DeveloperRiskProfile;
use crate::multi_offer_track_engine::MultiOfferState;

#[derive(Debug, Clone)]
pub struct ValuationConfidence {
    pub estimated_value_low: f64,
    pub estimated_value_high: f64,
    pub confidence_score: f64,           // 0.0 = very low confidence, 1.0 = high
    pub key_positive_factors: Vec<String>,
    pub key_risk_factors: Vec<String>,
    pub merciful_explanation: String,
    pub patsagi_notes: Vec<String>,
}

pub struct ValuationConfidenceScorer;

impl ValuationConfidenceScorer {
    /// Generates a valuation confidence assessment.
    /// This is intentionally conservative and context-rich.
    pub fn assess(
        property_type: &OntarioPropertyType,
        deal_type: &DealType,
        status: Option<&StatusCertificateAnalysis>,
        developer_risk: Option<&DeveloperRiskProfile>,
        multi_offer_state: Option<&MultiOfferState>,
        base_avm_estimate: Option<f64>, // Optional external AVM signal
    ) -> ValuationConfidence {
        let mut positive_factors = vec![];
        let mut risk_factors = vec![];
        let mut patsagi_notes = vec![];
        let mut confidence: f64 = 0.65; // baseline

        // Start with multi-offer data if available (strongest real-time signal)
        let (low, high) = if let Some(state) = multi_offer_state {
            if state.offers.len() >= 2 {
                positive_factors.push("Multiple active offers provide real-time market discovery".to_string());
                confidence += 0.15;

                let min_p = state.offers.values().map(|o| o.price).fold(f64::INFINITY, f64::min);
                let max_p = state.offers.values().map(|o| o.price).fold(f64::NEG_INFINITY, f64::max);
                (min_p * 0.97, max_p * 1.03) // small buffer
            } else {
                (0.0, 0.0)
            }
        } else {
            (0.0, 0.0)
        };

        // Status Certificate impact
        if let Some(s) = status {
            if s.special_assessments_pending || s.litigation_risk {
                risk_factors.push("Status Certificate shows special assessments or litigation risk".to_string());
                confidence -= 0.20;
            } else if s.overall_risk_level == "Low" {
                positive_factors.push("Clean Status Certificate supports valuation stability".to_string());
                confidence += 0.08;
            }
        }

        // Developer risk impact (especially pre-construction)
        if let Some(dev) = developer_risk {
            if dev.overall_risk_score > 0.6 {
                risk_factors.push(format!("Elevated developer risk (score {:.2})", dev.overall_risk_score));
                confidence -= 0.15;
            } else if dev.overall_risk_score < 0.35 {
                positive_factors.push("Low developer risk supports valuation confidence".to_string());
            }
        }

        // Deal type adjustments
        match deal_type {
            DealType::FamilyTransfer => {
                patsagi_notes.push("Family transfer detected. Valuation should consider long-term family impact and fairness beyond pure market comps.".to_string());
                confidence -= 0.05; // slightly more conservative
            }
            DealType::PreConstruction => {
                risk_factors.push("Pre-construction valuation carries completion and deposit protection uncertainty".to_string());
            }
            _ => {}
        }

        // External AVM signal (if provided)
        let final_low = if low > 0.0 { low } else { base_avm_estimate.unwrap_or(0.0) * 0.92 };
        let final_high = if high > 0.0 { high } else { base_avm_estimate.unwrap_or(0.0) * 1.08 };

        if base_avm_estimate.is_some() {
            positive_factors.push("External AVM signal incorporated with confidence adjustment".to_string());
        }

        // Final confidence clamping
        let confidence_score = confidence.clamp(0.25, 0.92);

        let merciful_explanation = if risk_factors.is_empty() {
            "Valuation confidence is reasonably strong based on available market and risk signals. Multiple data points align.".to_string()
        } else {
            format!(
                "Valuation has moderate confidence. Key risks noted: {}. Proceed with additional due diligence and lawyer review.",
                risk_factors.join("; ")
            )
        };

        ValuationConfidence {
            estimated_value_low: final_low,
            estimated_value_high: final_high,
            confidence_score,
            key_positive_factors: positive_factors,
            key_risk_factors: risk_factors,
            merciful_explanation,
            patsagi_notes,
        }
    }
}
