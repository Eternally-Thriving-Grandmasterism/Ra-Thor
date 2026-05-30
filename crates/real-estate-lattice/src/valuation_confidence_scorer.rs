//! Valuation Confidence Scorer for Real Estate Lattice
//!
//! Generates context-aware valuation confidence scores and merciful explanations
//! by combining signals from PropertyType, DealType, StatusCertificateAnalysis,
//! DeveloperRisk, Multi-Offer dynamics, **and external AVM signals**.
//!
//! Supports ingestion of external Automated Valuation Models (e.g. Teranet, HouseCanary, etc.)
//! while using our internal risk and real-time offer data as a critical reality check.
//!
//! This moves us toward a true Hybrid AVM that is more robust than pure external models.
//!
//! **Design Principles**:
//! - External AVM signals are respected but never blindly trusted
//! - Internal signals (especially Status Certificate + Multi-Offer pressure) act as overrides or dampeners
//! - Clear explanation when external and internal views diverge
//! - Mercy-first language in all outputs
//!
//! Part of the evolving Hybrid Valuation capability in the Real Estate Lattice.

use crate::property_type_classifier::OntarioPropertyType;
use crate::deal_type_classifier::DealType;
use crate::status_certificate_analyzer::StatusCertificateAnalysis;
use crate::developer_risk_engine::DeveloperRiskProfile;
use crate::multi_offer_track_engine::MultiOfferState;

/// Represents a signal coming from an external Automated Valuation Model provider.
#[derive(Debug, Clone)]
pub struct ExternalAvmSignal {
    pub provider: String,
    pub estimated_value: f64,
    pub provider_confidence: Option<f64>, // 0.0 - 1.0 if the provider reports it
    pub as_of: Option<String>,            // Optional timestamp or "as of" date
}

#[derive(Debug, Clone)]
pub struct ValuationConfidence {
    pub estimated_value_low: f64,
    pub estimated_value_high: f64,
    pub confidence_score: f64,
    pub key_positive_factors: Vec<String>,
    pub key_risk_factors: Vec<String>,
    pub merciful_explanation: String,
    pub patsagi_notes: Vec<String>,
}

pub struct ValuationConfidenceScorer;

impl ValuationConfidenceScorer {
    /// Generates a valuation confidence assessment, optionally incorporating an external AVM signal.
    pub fn assess(
        property_type: &OntarioPropertyType,
        deal_type: &DealType,
        status: Option<&StatusCertificateAnalysis>,
        developer_risk: Option<&DeveloperRiskProfile>,
        multi_offer_state: Option<&MultiOfferState>,
        external_avm: Option<&ExternalAvmSignal>,
    ) -> ValuationConfidence {
        let mut positive_factors = vec![];
        let mut risk_factors = vec![];
        let mut patsagi_notes = vec![];
        let mut confidence: f64 = 0.60;

        // === External AVM Signal Ingestion ===
        let mut base_value: Option<f64> = None;

        if let Some(avm) = external_avm {
            base_value = Some(avm.estimated_value);
            positive_factors.push(format!("External AVM signal from {} ingested", avm.provider));

            if let Some(avm_conf) = avm.provider_confidence {
                confidence += (avm_conf - 0.5) * 0.2; // reward or penalize based on provider confidence
            } else {
                confidence += 0.05; // slight boost just for having an external signal
            }
        }

        // === Multi-Offer Data (strongest real-time signal) ===
        let (low, high) = if let Some(state) = multi_offer_state {
            if state.offers.len() >= 2 {
                positive_factors.push("Multiple active offers provide strong real-time market discovery".to_string());
                confidence += 0.18;

                let min_p = state.offers.values().map(|o| o.price).fold(f64::INFINITY, f64::min);
                let max_p = state.offers.values().map(|o| o.price).fold(f64::NEG_INFINITY, f64::max);
                (min_p * 0.96, max_p * 1.04)
            } else {
                (base_value.unwrap_or(0.0) * 0.90, base_value.unwrap_or(0.0) * 1.10)
            }
        } else if let Some(val) = base_value {
            (val * 0.92, val * 1.08)
        } else {
            (0.0, 0.0)
        };

        // === Status Certificate Impact ===
        if let Some(s) = status {
            if s.special_assessments_pending || s.litigation_risk {
                risk_factors.push("Status Certificate indicates special assessments or litigation risk".to_string());
                confidence -= 0.22;
            } else if s.overall_risk_level == "Low" {
                positive_factors.push("Clean Status Certificate supports valuation stability".to_string());
                confidence += 0.10;
            }
        }

        // === Developer Risk Impact ===
        if let Some(dev) = developer_risk {
            if dev.overall_risk_score > 0.6 {
                risk_factors.push(format!("Elevated developer/pre-construction risk (score {:.2})", dev.overall_risk_score));
                confidence -= 0.18;
            }
        }

        // === Deal Type Adjustments ===
        match deal_type {
            DealType::FamilyTransfer => {
                patsagi_notes.push("Family transfer context detected. Valuation considers long-term relational impact beyond pure market signals.".to_string());
                confidence -= 0.06;
            }
            DealType::PreConstruction => {
                risk_factors.push("Pre-construction elements introduce completion and deposit protection uncertainty".to_string());
            }
            _ => {}
        }

        // === Divergence Detection between External AVM and Internal Signals ===
        if let (Some(avm), Some(state)) = (external_avm, multi_offer_state) {
            if state.offers.len() >= 2 {
                let offer_high = state.offers.values().map(|o| o.price).fold(f64::NEG_INFINITY, f64::max);
                let divergence = (avm.estimated_value - offer_high).abs() / offer_high.max(1.0);

                if divergence > 0.12 {
                    risk_factors.push(format!(
                        "Significant divergence ({:.1}%) between external AVM (${:.0}) and current highest offer (${:.0})",
                        divergence * 100.0,
                        avm.estimated_value,
                        offer_high
                    ));
                    confidence -= 0.12;
                }
            }
        }

        let confidence_score = confidence.clamp(0.20, 0.93);

        let merciful_explanation = if risk_factors.is_empty() {
            "Valuation confidence is solid. External signals and internal risk data are reasonably aligned.".to_string()
        } else {
            format!(
                "Valuation confidence is tempered. Notable risks or divergences detected: {}. Additional due diligence recommended.",
                risk_factors.join("; ")
            )
        };

        ValuationConfidence {
            estimated_value_low: low,
            estimated_value_high: high,
            confidence_score,
            key_positive_factors: positive_factors,
            key_risk_factors: risk_factors,
            merciful_explanation,
            patsagi_notes,
        }
    }
}
