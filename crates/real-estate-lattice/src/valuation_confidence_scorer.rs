//! Valuation Confidence Scorer for Real Estate Lattice
//!
//! Generates context-aware valuation confidence scores and merciful explanations.
//! Supports external AVM signal ingestion + simple in-memory caching with TTL.
//!
//! The cache helps avoid repeated expensive or rate-limited calls to external AVM providers
//! while still allowing fresh data when needed.
//!
//! **Caching Strategy**:
//! - In-memory HashMap with property key + timestamp
//! - Configurable TTL (default 24 hours)
//! - Simple and dependency-free for now
//! - Easy to later replace with Redis, Sled, or persistent store
//!
//! Part of the Hybrid Valuation system.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::property_type_classifier::OntarioPropertyType;
use crate::deal_type_classifier::DealType;
use crate::status_certificate_analyzer::StatusCertificateAnalysis;
use crate::developer_risk_engine::DeveloperRiskProfile;
use crate::multi_offer_track_engine::MultiOfferState;

/// Signal from an external Automated Valuation Model provider.
#[derive(Debug, Clone)]
pub struct ExternalAvmSignal {
    pub provider: String,
    pub estimated_value: f64,
    pub provider_confidence: Option<f64>,
    pub as_of: Option<String>,
}

/// Simple in-memory cache for external AVM signals.
/// Keyed by property identifier (e.g. PIN, normalized address, or MLS number).
#[derive(Debug, Default)]
pub struct AvmCache {
    entries: HashMap<String, (ExternalAvmSignal, u64)>, // (signal, timestamp)
    ttl_seconds: u64,
}

impl AvmCache {
    pub fn new(ttl_seconds: u64) -> Self {
        Self {
            entries: HashMap::new(),
            ttl_seconds,
        }
    }

    pub fn with_default_ttl() -> Self {
        Self::new(86_400) // 24 hours
    }

    /// Returns a cached signal if it exists and is still fresh.
    pub fn get(&self, key: &str) -> Option<ExternalAvmSignal> {
        if let Some((signal, timestamp)) = self.entries.get(key) {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            if now - *timestamp <= self.ttl_seconds {
                return Some(signal.clone());
            }
        }
        None
    }

    /// Inserts or updates a cached AVM signal.
    pub fn insert(&mut self, key: String, signal: ExternalAvmSignal) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.entries.insert(key, (signal, now));
    }

    /// Clears expired entries (optional maintenance).
    pub fn cleanup_expired(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.entries.retain(|_, (_, ts)| now - *ts <= self.ttl_seconds);
    }
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
    /// Assess valuation confidence, optionally using a cached external AVM signal.
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

        let mut base_value: Option<f64> = None;

        if let Some(avm) = external_avm {
            base_value = Some(avm.estimated_value);
            positive_factors.push(format!("External AVM signal from {} ingested", avm.provider));

            if let Some(avm_conf) = avm.provider_confidence {
                confidence += (avm_conf - 0.5) * 0.2;
            } else {
                confidence += 0.05;
            }
        }

        let (low, high) = if let Some(state) = multi_offer_state {
            if state.offers.len() >= 2 {
                positive_factors.push("Multiple active offers provide strong real-time market discovery".to_string());
                confidence += 0.18;

                let min_p = state.offers.values().map(|o| o.price).fold(f64::INFINITY, f64::min);
                let max_p = state.offers.values().map(|o| o.price).fold(f64::NEG_INFINITY, f64::max);
                (min_p * 0.96, max_p * 1.04)
            } else if let Some(val) = base_value {
                (val * 0.90, val * 1.10)
            } else {
                (0.0, 0.0)
            }
        } else if let Some(val) = base_value {
            (val * 0.92, val * 1.08)
        } else {
            (0.0, 0.0)
        };

        if let Some(s) = status {
            if s.special_assessments_pending || s.litigation_risk {
                risk_factors.push("Status Certificate indicates special assessments or litigation risk".to_string());
                confidence -= 0.22;
            } else if s.overall_risk_level == "Low" {
                positive_factors.push("Clean Status Certificate supports valuation stability".to_string());
                confidence += 0.10;
            }
        }

        if let Some(dev) = developer_risk {
            if dev.overall_risk_score > 0.6 {
                risk_factors.push(format!("Elevated developer risk (score {:.2})", dev.overall_risk_score));
                confidence -= 0.18;
            }
        }

        match deal_type {
            DealType::FamilyTransfer => {
                patsagi_notes.push("Family transfer context detected. Valuation considers long-term relational impact.".to_string());
                confidence -= 0.06;
            }
            DealType::PreConstruction => {
                risk_factors.push("Pre-construction uncertainty present".to_string());
            }
            _ => {}
        }

        if let (Some(avm), Some(state)) = (external_avm, multi_offer_state) {
            if state.offers.len() >= 2 {
                let offer_high = state.offers.values().map(|o| o.price).fold(f64::NEG_INFINITY, f64::max);
                let divergence = (avm.estimated_value - offer_high).abs() / offer_high.max(1.0);

                if divergence > 0.12 {
                    risk_factors.push(format!(
                        "Divergence ({:.1}%) between external AVM and current highest offer",
                        divergence * 100.0
                    ));
                    confidence -= 0.12;
                }
            }
        }

        let confidence_score = confidence.clamp(0.20, 0.93);

        let merciful_explanation = if risk_factors.is_empty() {
            "Valuation confidence is solid based on available signals.".to_string()
        } else {
            format!("Valuation has tempered confidence due to: {}. Recommend further review.", risk_factors.join("; "))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avm_cache_basic() {
        let mut cache = AvmCache::with_default_ttl();
        let signal = ExternalAvmSignal {
            provider: "TestAVM".to_string(),
            estimated_value: 875_000.0,
            provider_confidence: Some(0.82),
            as_of: None,
        };

        cache.insert("PIN-123456".to_string(), signal.clone());
        let cached = cache.get("PIN-123456");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().estimated_value, 875_000.0);
    }

    #[test]
    fn test_avm_cache_expires() {
        let mut cache = AvmCache::new(1); // 1 second TTL
        let signal = ExternalAvmSignal {
            provider: "ShortLived".to_string(),
            estimated_value: 900_000.0,
            provider_confidence: None,
            as_of: None,
        };

        cache.insert("PIN-EXP".to_string(), signal);
        std::thread::sleep(std::time::Duration::from_secs(2));
        assert!(cache.get("PIN-EXP").is_none());
    }
}
