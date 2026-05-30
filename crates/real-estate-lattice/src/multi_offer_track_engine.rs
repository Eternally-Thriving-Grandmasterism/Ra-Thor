//! Multi-Offer Track Engine for Real Estate Lattice
//!
//! Production-grade management of multiple competing offers on the same property.
//! Handles escalation, bully offers, fairness tracking, and professional strategy recommendations.
//!
//! **Ontario Context**:
//! - Common in hot markets (especially GTA)
//! - Requires careful handling of escalation clauses, bully offers, and disclosure rules
//! - Fairness and transparency protect all parties and reduce post-deal disputes
//!
//! **Privacy & Mercy**:
//! - Tracks offer metadata only (no full financial PII unless explicitly passed)
//! - Provides balanced strategy notes instead of aggressive tactics
//! - PATSAGi guidance for high-conflict or multi-party situations
//!
//! Integrates with DealTypeClassifier and OfferPackageValidator.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Offer {
    pub buyer_id: u64,
    pub price: f64,
    pub conditions: Vec<String>,
    pub escalation_clause: bool,
    pub is_bully: bool,
}

#[derive(Debug, Clone)]
pub struct MultiOfferState {
    pub offers: HashMap<u64, Offer>,
    pub highest_price: f64,
    pub active_escalations: u32,
    pub fairness_notes: Vec<String>,
    pub recommended_strategy: String,
}

pub struct MultiOfferTrackEngine;

impl MultiOfferTrackEngine {
    pub fn new() -> Self {
        Self
    }

    /// Registers or updates an offer in the tracking system.
    pub fn register_offer(&self, state: &mut MultiOfferState, offer: Offer) {
        if offer.price > state.highest_price {
            state.highest_price = offer.price;
        }
        if offer.escalation_clause {
            state.active_escalations += 1;
        }
        if offer.is_bully {
            state.fairness_notes.push(format!("Bully offer detected from buyer {}. Consider disclosure and fairness review.", offer.buyer_id));
        }
        state.offers.insert(offer.buyer_id, offer);
    }

    /// Analyzes current multi-offer situation and produces strategy + fairness guidance.
    pub fn analyze_and_recommend(&self, state: &MultiOfferState) -> (Vec<String>, String) {
        let mut notes = state.fairness_notes.clone();
        let mut strategy = String::from("Standard multi-offer protocol: Present all offers transparently to seller. ");

        if state.active_escalations > 0 {
            notes.push("Escalation clauses active. Ensure all escalation terms are clearly disclosed and capped where appropriate.".to_string());
            strategy.push_str("Review escalation caps and triggers with seller. ");
        }

        if state.offers.len() > 2 {
            notes.push("Multiple competing offers present. Maintain strict fairness in communication and timing.".to_string());
            strategy.push_str("Consider setting a deadline for best-and-final offers to reduce prolonged negotiation stress.");
        }

        if notes.is_empty() {
            notes.push("No immediate fairness concerns detected. Proceed with standard best practices.".to_string());
        }

        (notes, strategy)
    }
}
