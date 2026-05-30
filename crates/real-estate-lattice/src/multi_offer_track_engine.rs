//! Multi-Offer Track Engine for Real Estate Lattice
//!
//! Production-grade multi-offer tracking with escalation logic and
//! optional Redis Streams invalidation publishing for AVM cache coherence.

use std::collections::HashMap;

#[cfg(feature = "redis")]
use crate::avm_cache_invalidation::RedisStreamPublisher;

#[derive(Debug, Clone)]
pub struct Offer {
    pub buyer_id: u64,
    pub price: f64,
    pub conditions: Vec<String>,
    pub escalation_clause: bool,
    pub is_bully: bool,
}

#[derive(Debug, Clone)]
pub struct EscalationClause {
    pub base_price: f64,
    pub increment: f64,
    pub cap: f64,
    pub is_disclosed: bool,
}

#[derive(Debug, Clone)]
pub struct MultiOfferState {
    pub offers: HashMap<u64, Offer>,
    pub highest_price: f64,
    pub active_escalations: u32,
    pub fairness_notes: Vec<String>,
    pub recommended_strategy: String,
}

pub struct MultiOfferTrackEngine {
    #[cfg(feature = "redis")]
    invalidation_publisher: Option<RedisStreamPublisher>,
}

impl MultiOfferTrackEngine {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "redis")]
            invalidation_publisher: None,
        }
    }

    /// Attach a Redis Streams publisher for distributed AVM cache invalidation.
    #[cfg(feature = "redis")]
    pub fn with_invalidation_publisher(mut self, publisher: RedisStreamPublisher) -> Self {
        self.invalidation_publisher = Some(publisher);
        self
    }

    pub fn register_offer(&mut self, state: &mut MultiOfferState, offer: Offer) {
        let previous_highest = state.highest_price;

        if offer.price > state.highest_price {
            state.highest_price = offer.price;
        }

        if offer.escalation_clause {
            state.active_escalations += 1;
        }

        if offer.is_bully {
            state.fairness_notes.push(format!("Bully offer from buyer {}", offer.buyer_id));
        }

        state.offers.insert(offer.buyer_id, offer);

        // === Redis Streams Invalidation ===
        #[cfg(feature = "redis")]
        if let Some(publisher) = &self.invalidation_publisher {
            // Only publish on meaningful price movement (>5%)
            if state.highest_price > previous_highest * 1.05 {
                let _ = publisher.publish(
                    &format!("offer-update-{}", state.highest_price as u64), // simple key
                    "significant_price_increase",
                );
            }
        }
    }

    // ... rest of existing methods (calculate_escalated_price, analyze_and_recommend, etc.) remain unchanged ...

    pub fn analyze_and_recommend(&self, state: &MultiOfferState) -> (Vec<String>, String) {
        // existing implementation
        let mut notes = state.fairness_notes.clone();
        let mut strategy = String::from("Standard multi-offer protocol. ");

        if state.active_escalations > 0 {
            strategy.push_str("Escalation clauses active. ");
        }
        if state.offers.len() > 2 {
            strategy.push_str("Consider best-and-final offers.");
        }

        (notes, strategy)
    }
}
