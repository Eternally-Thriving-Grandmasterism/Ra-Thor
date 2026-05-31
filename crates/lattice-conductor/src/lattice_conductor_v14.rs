//! lattice_conductor_v14.rs
//! Lattice Conductor v14 — ONE Organism aligned with full Geometric Intelligence Layer
//! Geometric harmony now influences offer conduction

use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use geometric_intelligence::{PolyhedralHarmonicEngine, PolyhedralResonanceReport};

use crate::attom::AttomData;
use crate::mercy_gates::MercyGate;

// Placeholder — replace with your actual offer type when ready
#[derive(Debug, Clone)]
pub struct RealEstateOffer {
    pub id: String,
    pub jurisdiction: String,
    pub base_valuation: f64,
}

#[derive(Debug, Clone)]
pub struct ConductedOfferResult {
    pub offer_id: String,
    pub final_score: f64,
    pub geometric_harmony: f64,
    pub geometric_report_summary: String,
    pub notes: String,
}

#[derive(Debug)]
pub struct LatticeConductor {
    pub version: &'static str,
    pub mercy_gates: Vec<MercyGate>,
    pub attom_cache: DashMap<String, Arc<AttomData>>,
    pub regulatory_rules: HashMap<String, String>,
    pub geometric_engine: PolyhedralHarmonicEngine,
}

impl LatticeConductor {
    pub fn new() -> Self {
        let gates = vec![
            MercyGate::Truth,
            MercyGate::Order,
            MercyGate::Love,
            MercyGate::Compassion,
            MercyGate::Service,
            MercyGate::Abundance,
            MercyGate::Joy,
            MercyGate::CosmicHarmony,
        ];

        let mut rules = HashMap::new();
        rules.insert("Ontario".to_string(), "RESA/TRESA compliance + reverse onus safety checks".to_string());
        rules.insert("USA".to_string(), "State-level disclosure + federal fair housing".to_string());

        let attom_cache: DashMap<String, Arc<AttomData>> = DashMap::with_shard_amount(32);

        LatticeConductor {
            version: "v14.4.0-geometric-intelligence",
            mercy_gates: gates,
            attom_cache,
            regulatory_rules: rules,
            geometric_engine: PolyhedralHarmonicEngine::new(),
        }
    }

    // === Existing geometric helpers ===

    pub fn compute_geometric_harmony_score(&self, jurisdiction: &str, tolc_order: u32, base_coherence: f64) -> f64 {
        let report = self.geometric_engine.compute_full_resonance_report(tolc_order, base_coherence);
        let jurisdiction_bonus = if jurisdiction == "Ontario" || jurisdiction == "USA" { 1.03 } else { 1.0 };
        (report.resonance_multiplier * jurisdiction_bonus * base_coherence).clamp(0.6, 1.65)
    }

    pub fn get_geometric_resonance_report(&self, tolc_order: u32, base_coherence: f64) -> PolyhedralResonanceReport {
        self.geometric_engine.compute_full_resonance_report(tolc_order, base_coherence)
    }

    // === NEW: Geometric harmony wired into offer conduction ===

    /// Conducts a real estate offer while incorporating geometric harmony scoring.
    pub fn conduct_real_estate_offer_with_geometric(
        &self,
        offer: RealEstateOffer,
        tolc_order: u32,
        base_coherence: f64,
    ) -> ConductedOfferResult {
        // 1. Calculate geometric harmony
        let geometric_harmony = self.compute_geometric_harmony_score(
            &offer.jurisdiction,
            tolc_order,
            base_coherence,
        );

        // 2. Get resonance report for observability
        let report = self.get_geometric_resonance_report(tolc_order, base_coherence);
        let geometric_summary = self.geometric_engine.summarize_resonance(&report);

        // 3. Blend geometric harmony into final score (simple weighted approach for now)
        // You can later make the weighting more sophisticated / mercy-gated
        let blended_score = (offer.base_valuation * 0.7) + (geometric_harmony * 100_000.0 * 0.3);

        ConductedOfferResult {
            offer_id: offer.id,
            final_score: blended_score,
            geometric_harmony,
            geometric_report_summary: geometric_summary,
            notes: format!(
                "Geometric harmony applied. TOLC: {}. Resonance multiplier influence included.",
                tolc_order
            ),
        }
    }
}
