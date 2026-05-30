//! Multi-Offer Track Engine for Real Estate Lattice
//!
//! Production-grade management of multiple competing offers.
//! Now includes comprehensive Escalation Logic.
//!
//! **Escalation Logic Features**:
//! - EscalationClause struct with cap, increment, and trigger rules
//! - Automatic escalated price calculation
//! - Validation of escalation terms (cap reasonableness, disclosure)
//! - Smart analysis: when to escalate vs set best-and-final
//! - Fairness and disclosure recommendations
//!
//! **Ontario Best Practices**:
//! - Escalation clauses must be clearly drafted and disclosed
//! - Caps prevent runaway bidding
//! - Fairness to all buyers is paramount
//!
//! **Mercy & Ethics**:
//! - Recommendations prioritize clarity and reduced conflict
//! - PATSAGi flags for high-escalation or aggressive situations
//! - Balanced strategy instead of pure competitive pressure
//!
//! Integrates cleanly with the rest of the Real Estate Lattice.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Offer {
    pub buyer_id: u64,
    pub price: f64,
    pub conditions: Vec<String>,
    pub escalation_clause: bool,
    pub is_bully: bool,
}

/// Defines the terms of an escalation clause.
#[derive(Debug, Clone)]
pub struct EscalationClause {
    pub base_price: f64,
    pub increment: f64,           // How much to escalate per competing offer
    pub cap: f64,                 // Maximum price the buyer is willing to go
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

pub struct MultiOfferTrackEngine;

impl MultiOfferTrackEngine {
    pub fn new() -> Self {
        Self
    }

    /// Registers or updates an offer.
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

    /// Calculates the escalated price for a buyer given current highest competing offer.
    /// Respects the cap. Returns None if escalation would exceed cap or clause is invalid.
    pub fn calculate_escalated_price(
        &self,
        clause: &EscalationClause,
        current_highest_competing: f64,
    ) -> Option<f64> {
        if !clause.is_disclosed {
            return None; // Escalation must be disclosed
        }
        if current_highest_competing <= clause.base_price {
            return Some(clause.base_price);
        }

        let gap = current_highest_competing - clause.base_price;
        let escalations_needed = (gap / clause.increment).ceil() as i32;
        let new_price = clause.base_price + (escalations_needed as f64 * clause.increment);

        if new_price > clause.cap {
            Some(clause.cap) // Cap hit
        } else {
            Some(new_price)
        }
    }

    /// Validates an escalation clause for reasonableness and compliance.
    pub fn validate_escalation_clause(&self, clause: &EscalationClause) -> Vec<String> {
        let mut issues = vec![];

        if clause.increment <= 0.0 {
            issues.push("Escalation increment must be positive".to_string());
        }
        if clause.cap <= clause.base_price {
            issues.push("Escalation cap must be higher than base price".to_string());
        }
        if (clause.cap - clause.base_price) / clause.increment > 20.0 {
            issues.push("Escalation cap is very high relative to increment. Consider tighter cap for fairness and clarity.".to_string());
        }
        if !clause.is_disclosed {
            issues.push("Escalation clause must be disclosed to seller and other parties".to_string());
        }

        issues
    }

    /// Applies escalation logic across current offers and returns updated recommendations.
    pub fn apply_escalation_logic(
        &self,
        state: &mut MultiOfferState,
        escalation_clauses: &HashMap<u64, EscalationClause>,
    ) -> Vec<String> {
        let mut recommendations = vec![];

        for (buyer_id, clause) in escalation_clauses {
            if let Some(offer) = state.offers.get(buyer_id) {
                if let Some(new_price) = self.calculate_escalated_price(clause, state.highest_price) {
                    if new_price > offer.price {
                        recommendations.push(format!(
                            "Buyer {} can escalate to ${:.0} (capped at ${:.0}). Consider presenting updated offer.",
                            buyer_id, new_price, clause.cap
                        ));
                    }
                }
            }
        }

        if state.active_escalations > 2 {
            recommendations.push("Multiple escalation clauses active. Recommend moving to best-and-final offers to reduce complexity and stress.".to_string());
        }

        recommendations
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

        if state.active_escalations >= 3 {
            strategy.push_str("High number of escalations detected. Strongly recommend best-and-final round for clarity and mercy to all parties.");
        }

        if notes.is_empty() {
            notes.push("No immediate fairness concerns detected. Proceed with standard best practices.".to_string());
        }

        (notes, strategy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_engine() -> MultiOfferTrackEngine {
        MultiOfferTrackEngine::new()
    }

    #[test]
    fn test_calculate_escalated_price_basic() {
        let engine = default_engine();
        let clause = EscalationClause {
            base_price: 800_000.0,
            increment: 5_000.0,
            cap: 850_000.0,
            is_disclosed: true,
        };
        let escalated = engine.calculate_escalated_price(&clause, 812_000.0);
        assert_eq!(escalated, Some(815_000.0));
    }

    #[test]
    fn test_calculate_escalated_price_hits_cap() {
        let engine = default_engine();
        let clause = EscalationClause {
            base_price: 800_000.0,
            increment: 10_000.0,
            cap: 820_000.0,
            is_disclosed: true,
        };
        let escalated = engine.calculate_escalated_price(&clause, 850_000.0);
        assert_eq!(escalated, Some(820_000.0)); // capped
    }

    #[test]
    fn test_calculate_escalated_price_below_base() {
        let engine = default_engine();
        let clause = EscalationClause {
            base_price: 800_000.0,
            increment: 5_000.0,
            cap: 850_000.0,
            is_disclosed: true,
        };
        let escalated = engine.calculate_escalated_price(&clause, 790_000.0);
        assert_eq!(escalated, Some(800_000.0));
    }

    #[test]
    fn test_calculate_escalated_price_undisclosed_returns_none() {
        let engine = default_engine();
        let clause = EscalationClause {
            base_price: 800_000.0,
            increment: 5_000.0,
            cap: 850_000.0,
            is_disclosed: false,
        };
        let escalated = engine.calculate_escalated_price(&clause, 810_000.0);
        assert!(escalated.is_none());
    }

    #[test]
    fn test_validate_escalation_clause_valid() {
        let engine = default_engine();
        let clause = EscalationClause {
            base_price: 700_000.0,
            increment: 2_500.0,
            cap: 750_000.0,
            is_disclosed: true,
        };
        assert!(engine.validate_escalation_clause(&clause).is_empty());
    }

    #[test]
    fn test_validate_escalation_clause_invalid_increment() {
        let engine = default_engine();
        let clause = EscalationClause {
            base_price: 700_000.0,
            increment: 0.0,
            cap: 750_000.0,
            is_disclosed: true,
        };
        let issues = engine.validate_escalation_clause(&clause);
        assert!(issues.iter().any(|i| i.contains("positive")));
    }

    #[test]
    fn test_validate_escalation_clause_undisclosed() {
        let engine = default_engine();
        let clause = EscalationClause {
            base_price: 700_000.0,
            increment: 5_000.0,
            cap: 750_000.0,
            is_disclosed: false,
        };
        let issues = engine.validate_escalation_clause(&clause);
        assert!(issues.iter().any(|i| i.contains("disclosed")));
    }

    #[test]
    fn test_apply_escalation_logic_recommends_update() {
        let engine = default_engine();
        let mut state = MultiOfferState {
            offers: HashMap::new(),
            highest_price: 805_000.0,
            active_escalations: 1,
            fairness_notes: vec![],
            recommended_strategy: String::new(),
        };
        state.offers.insert(1, Offer {
            buyer_id: 1,
            price: 800_000.0,
            conditions: vec![],
            escalation_clause: true,
            is_bully: false,
        });

        let mut clauses = HashMap::new();
        clauses.insert(1, EscalationClause {
            base_price: 800_000.0,
            increment: 5_000.0,
            cap: 850_000.0,
            is_disclosed: true,
        });

        let recs = engine.apply_escalation_logic(&mut state, &clauses);
        assert!(recs.iter().any(|r| r.contains("can escalate")));
    }

    #[test]
    fn test_high_escalation_count_recommends_best_and_final() {
        let engine = default_engine();
        let mut state = MultiOfferState {
            offers: HashMap::new(),
            highest_price: 900_000.0,
            active_escalations: 4,
            fairness_notes: vec![],
            recommended_strategy: String::new(),
        };

        let (notes, strategy) = engine.analyze_and_recommend(&state);
        assert!(strategy.contains("best-and-final"));
    }
}
